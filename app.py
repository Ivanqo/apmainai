# app.py

import os
import asyncio
import time
import logging
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import boto3
import requests

best_checkpoint = "ner_model/checkpoint-15000"
os.makedirs(best_checkpoint, exist_ok=True)

# Сюда впиши ссылки на свои файлы из Object Storage
BASE_URL = "https://storage.yandexcloud.net/model-inf"
MODEL_FILES = [
    "pytorch_model.bin",
    "vocab.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json"
]

def download_from_bucket():
    for file in MODEL_FILES:
        local_path = os.path.join(best_checkpoint, file)
        if not os.path.exists(local_path):
            url = f"{BASE_URL}/{file}"
            logging.info(f"Скачиваю {file} из {url}")
            r = requests.get(url)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)

# -------------------------
# Логирование
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------------
# Класс модели — Блок 1
# -------------------------
class NERWithCRF(nn.Module): 
    def __init__(self, model_name, num_labels, lstm_hidden=256, lstm_layers=2, dropout_prob=0.3):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True,
            output_hidden_states=True
        )

        self.hidden_size = self.encoder.config.hidden_size
        self.num_hidden_layers = self.encoder.config.num_hidden_layers + 1
        self.num_used_layers = 4
        self.layer_weights = nn.Parameter(torch.ones(self.num_used_layers) / self.num_used_layers)

        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_prob if lstm_layers > 1 else 0
        )

        self.layer_norm = nn.LayerNorm(lstm_hidden * 2)
        self.dropout = nn.Dropout(dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, num_labels)
        )

        self.crf = CRF(num_labels, batch_first=True)
        self.class_weights = nn.Parameter(torch.ones(num_labels), requires_grad=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        device = input_ids.device
        attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states
        last_layers = torch.stack(hidden_states[-self.num_used_layers:])
        weighted_layers = (last_layers * self.layer_weights.view(-1, 1, 1, 1)).sum(0)

        lengths = attention_mask.sum(dim=1)
        packed_input = pack_padded_sequence(weighted_layers, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)

        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)

        emissions = self.classifier(lstm_out)
        mask = attention_mask.bool()

        if labels is not None:
            labels_clean = labels.clone()
            labels_clean[labels == -100] = 0

            crf_loss = -self.crf(emissions, labels_clean, mask=mask, reduction="mean")
            macro_f1_loss = self.macro_f1_loss(self.crf.decode(emissions, mask=mask), labels_clean, mask)
            loss = crf_loss + 0.3 * macro_f1_loss
            return {"loss": loss, "logits": emissions}
        else:
            predictions = self.crf.decode(emissions, mask=mask)
            return {"logits": emissions, "predictions": predictions}

    def macro_f1_loss(self, preds, labels, mask):
        eps = 1e-8
        macro_f1 = 0
        num_classes = self.classifier[-1].out_features

        preds_tensor = torch.zeros_like(labels, device=labels.device)
        for i, seq in enumerate(preds):
            preds_tensor[i, :len(seq)] = torch.tensor(seq, device=labels.device)

        for cls in range(1, num_classes):
            tp = ((preds_tensor == cls) & (labels == cls) & mask).sum().float()
            fp = ((preds_tensor == cls) & (labels != cls) & mask).sum().float()
            fn = ((preds_tensor != cls) & (labels == cls) & mask).sum().float()
            f1 = 2 * tp / (2 * tp + fp + fn + eps)
            macro_f1 += (1 - f1)

        return macro_f1 / (num_classes - 1)

# -------------------------
# Глобальные переменные — Блок 2
# -------------------------
label_list = ["O", "B-TYPE", "I-TYPE", "B-BRAND", "I-BRAND", 
              "B-VOLUME", "I-VOLUME", "B-PERCENT", "I-PERCENT"]
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

model = None
tokenizer = None
DEVICE = None

def load_model_sync():
    global model, tokenizer, DEVICE
    tokenizer = AutoTokenizer.from_pretrained(best_checkpoint)
    model = NERWithCRF(
        model_name="DeepPavlov/rubert-base-cased-conversational",
        num_labels=len(label_list)
    )
    state_dict = torch.load(os.path.join(best_checkpoint, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(state_dict)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)
    model.eval()

    logger.info("Модель с CRF успешно загружена 🚀")

# -------------------------
# Вспомогательные функции — Блок 3
# -------------------------
with open("brands_all.txt", "r", encoding="utf-8") as f:
    BRANDS = [line.strip().lower() for line in f if line.strip()]

BRANDS += ["coca-cola", "pepsi", "fanta", "sprite", "lipton", "lay's", "pringles", "kitkat", "oreo", "milka", "простоквашино", "домик в деревне", "valio", "danone", "добрый", "rich", "j7", "bonaqua", "святой источник"]

VOLUME_UNITS = {"г", "кг", "л", "мл", "шт", "гр", "грамм", "килограмм"}
PERCENT_UNITS = {"%", "проц", "процент"}

def validate_entity(entity_type: str, text: str) -> bool:
    text = text.strip().lower()
    if not text or len(text) == 1:
        return False

    if entity_type == "TYPE":
        return len(text) >= 3 and not text.isdigit()
    if entity_type == "BRAND":
        return text in BRANDS
    if entity_type == "VOLUME":
        return any(unit in text for unit in VOLUME_UNITS) and any(char.isdigit() for char in text)
    if entity_type == "PERCENT":
        return any(unit in text for unit in PERCENT_UNITS) and any(char.isdigit() for char in text)
    return False

def postprocess_annotations(text: str, raw_annotations: list) -> list:
    if not raw_annotations:
        return []
    merged = []
    prev_label = None
    prev_type = None
    prev_end = None
    for start_char, end_char, label in raw_annotations:
        if label == "O":
            merged.append((start_char, end_char, "O"))
            prev_label = None
            prev_type = None
            prev_end = None
            continue
        current_type = label.split("-", 1)[1]
        if prev_label and (prev_label.startswith("B-") or prev_label.startswith("I-")):
            if current_type == prev_type and start_char == prev_end + 1:
                merged.append((start_char, end_char, f"I-{current_type}"))
                prev_label = f"I-{current_type}"
                prev_end = end_char
                continue
        merged.append((start_char, end_char, label))
        prev_label = label
        prev_type = current_type
        prev_end = end_char
    return merged

def predict_annotations(text: str, keep_O: bool = True, fix_i_without_b: bool = True, agg_strategy: str = "first") -> list:
    if not text.strip():
        return []
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        is_split_into_words=False,
        return_special_tokens_mask=True,
        truncation=True,
        return_tensors="pt"
    )
    device = next(model.parameters()).device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offsets = encoding["offset_mapping"].squeeze(0)
    word_ids = encoding.word_ids(batch_index=0)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds_seq = outputs["predictions"][0]

    word_to_token_idxs = {}
    for tok_idx, w_id in enumerate(word_ids):
        if w_id is None:
            continue
        word_to_token_idxs.setdefault(w_id, []).append(tok_idx)

    raw_annotations = []
    prev_word_label = None
    for word_idx, token_idxs in word_to_token_idxs.items():
        token_labels = [preds_seq[tok_idx] for tok_idx in token_idxs]
        if agg_strategy == "first":
            word_label_id = token_labels[0]
        else:
            word_label_id = Counter(token_labels).most_common(1)[0][0]
        word_label = id2label[word_label_id]
        if fix_i_without_b and word_label.startswith("I-") and (prev_word_label is None or prev_word_label[2:] != word_label[2:]):
            word_label = "B-" + word_label[2:]
        prev_word_label = word_label
        if not keep_O and word_label == "O":
            continue
        start_char = offsets[token_idxs[0]][0].item()
        end_char = offsets[token_idxs[-1]][1].item()
        raw_annotations.append((start_char, end_char, word_label))
    return postprocess_annotations(text, raw_annotations)

# -------------------------
# FastAPI приложение
# -------------------------
app = FastAPI(title="NER BIO API", version="1.0")

class PredictRequest(BaseModel):
    input: str

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_model_sync)

@app.post("/api/predict")
async def predict(req: PredictRequest, request: Request):
    start_time = time.time()
    text = req.input
    if not text.strip():
        return []

    loop = asyncio.get_event_loop()
    annotations = await loop.run_in_executor(None, predict_annotations, text)

    elapsed = time.time() - start_time
    logger.info(f"Обработка запроса заняла {elapsed:.4f} секунд")
    return [
        {"start_index": start, "end_index": end, "entity": label}
        for start, end, label in annotations
        if label != "O"
    ]


uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
