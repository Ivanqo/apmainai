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

def download_from_bucket(best_checkpoint):
    logging.info(f"Проверяю и создаю директорию: {best_checkpoint}")
    os.makedirs(best_checkpoint, exist_ok=True)

    missing_files = []
    for file in MODEL_FILES:
        local_path = os.path.join(best_checkpoint, file)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            logging.info(f"Файл уже существует: {file}")
            continue
        missing_files.append(file)

    if not missing_files:
        logging.info("Все файлы уже присутствуют, скачивание не требуется ✅")
        return

    for file in missing_files:
        url = f"{BASE_URL}/{file}"
        logging.info(f"Начинаю скачивание {file} из {url}")
        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()

            total_size = int(r.headers.get("content-length", 0))
            logging.info(f"Размер файла {file}: {total_size} байт")

            local_path = os.path.join(best_checkpoint, file)
            with open(local_path, "wb") as f:
                downloaded = 0
                start_time = time.time()
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            elapsed = time.time() - start_time
                            speed = downloaded / (elapsed + 1e-6) / 1024
                            remaining = (total_size - downloaded) / (speed * 1024 + 1e-6)
                            if downloaded // total_size * 100 % 10 == 0:
                                logging.info(
                                    f"{file}: {percent:.2f}% скачано, "
                                    f"{downloaded}/{total_size} байт, скорость {speed:.2f} KB/s, "
                                    f"осталось ~{remaining:.1f} сек"
                                )

            if os.path.getsize(local_path) == 0:
                logging.error(f"Файл {local_path} пуст после загрузки!")
            else:
                logging.info(f"Файл {file} успешно загружен ✅")

        except Exception as e:
            logging.exception(f"Ошибка при скачивании {file}: {e}")



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

        # Загружаем энкодер
        self.encoder = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True,
            output_hidden_states=True
        )

        self.hidden_size = self.encoder.config.hidden_size
        self.num_hidden_layers = self.encoder.config.num_hidden_layers + 1  # включая embeddings

        # Будем использовать только последние 4 слоя
        self.num_used_layers = 4
        self.layer_weights = nn.Parameter(torch.ones(self.num_used_layers) / self.num_used_layers)

        # LSTM
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

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, num_labels)
        )

        # CRF слой
        self.crf = CRF(num_labels, batch_first=True)

        # Веса для классов (если будут нужны позже)
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

        hidden_states = outputs.hidden_states  # tuple из num_hidden_layers слоёв
        last_layers = torch.stack(hidden_states[-self.num_used_layers:])  # (4, batch, seq, hidden)

        # Взвешиваем только последние 4 слоя
        weighted_layers = (last_layers * self.layer_weights.view(-1, 1, 1, 1)).sum(0)

        # LSTM
        lengths = attention_mask.sum(dim=1)  # длины последовательностей
        packed_input = pack_padded_sequence(weighted_layers, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)

        # Паддинг до максимальной длины входа
        lstm_out, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=attention_mask.size(1)  # ключевой момент
        )

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

        for cls in range(1, num_classes):  # пропускаем O
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

d# -------------------------
# Исправленная загрузка модели — Блок 2
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
    try:
        logger.info(f"Проверяю содержимое директории: {best_checkpoint}")
        if os.path.exists(best_checkpoint):
            files = os.listdir(best_checkpoint)
            logger.info(f"Файлы в {best_checkpoint}: {files}")
        else:
            logger.error(f"Директория {best_checkpoint} не существует!")

        # КРИТИЧЕСКИ ВАЖНО: используем ту же модель, что и при обучении
        model_checkpoint = "DeepPavlov/rubert-base-cased-conversational"
        
        logger.info("Загружаю токенизатор...")
        tokenizer = AutoTokenizer.from_pretrained(best_checkpoint)
        logger.info("Токенизатор загружен ✅")

        logger.info("Создаю модель с теми же параметрами...")
        model = NERWithCRF(
            model_name=model_checkpoint,  # ТОЧНО ТАК ЖЕ КАК ПРИ ОБУЧЕНИИ
            num_labels=len(label_list),
        )
        logger.info("Архитектура модели создана ✅")

        model_path = os.path.join(best_checkpoint, "pytorch_model.bin")
        if not os.path.exists(model_path):
            logger.error(f"Файл весов модели не найден: {model_path}")
            return

        logger.info(f"Загружаю веса модели из {model_path}")
        state_dict = torch.load(model_path, map_location="cpu")
        
        # Важная проверка: сравни архитектуры
        model_state_keys = set(model.state_dict().keys())
        loaded_state_keys = set(state_dict.keys())
        
        if model_state_keys != loaded_state_keys:
            logger.warning(f"Расхождение в ключах модели!")
            missing = model_state_keys - loaded_state_keys
            extra = loaded_state_keys - model_state_keys
            if missing:
                logger.warning(f"Отсутствующие ключи: {missing}")
            if extra:
                logger.warning(f"Лишние ключи: {extra}")
        
        model.load_state_dict(state_dict, strict=False)  # strict=False для устойчивости
        logger.info("Веса модели загружены ✅")

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(DEVICE)
        model.eval()
        
        # Проверка устройства
        logger.info(f"Модель перемещена на устройство: {DEVICE}")
        logger.info(f"Первый параметр модели на: {next(model.parameters()).device}")
        
        logger.info("Модель с CRF успешно загружена 🚀")

    except Exception as e:
        logger.exception(f"Ошибка при загрузке модели: {e}")
        raise

# -------------------------
# Вспомогательные функции — Блок 3
# -------------------------
def aggregate_labels(subtok_labels: list) -> str:
    """
    Гибридная агрегация:
    1. Если все метки одинаковые → вернуть их.
    2. Если метки только B-/I- одного типа → вернуть B- этого типа.
    3. Если смесь с O → вернуть наиболее частую (majority).
    """
    if len(set(subtok_labels)) == 1:
        return subtok_labels[0]

    # Убираем O для проверки на один тип
    non_o = [l for l in subtok_labels if l != "O"]
    if non_o:
        types = {l.split("-", 1)[-1] for l in non_o}
        if len(types) == 1:
            # один тип, но разный префикс → начинаем с B-
            return "B-" + list(types)[0]

    # fallback → majority
    return Counter(subtok_labels).most_common(1)[0][0]


with open("/app/brands_all.txt", "r", encoding="utf-8") as f:
    BRANDS = [line.strip().lower() for line in f if line.strip()]
logger.info(f"{BRANDS}")
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
        return [(0, len(text), "O")]  # весь текст как O, если пусто

    merged = []
    prev_label = None
    prev_type = None
    prev_end = -1  # индекс конца предыдущего спана

    for start_char, end_char, label in raw_annotations:
        # Проверяем на "дыру" между предыдущим и текущим спаном
        if start_char > prev_end + 1:
            gap_start = prev_end + 1
            gap_end = start_char - 1
            gap_text = text[gap_start:gap_end + 1].strip()

            # 1. Если дыра очень маленькая (< 3 символов) → присоединяем к соседям
            if len(gap_text) <= 2:
                if merged and prev_label and prev_label != "O":
                    # расширяем предыдущую сущность
                    last_start, last_end, last_label = merged[-1]
                    merged[-1] = (last_start, gap_end, last_label)
                else:
                    # если после будет сущность → расширим её
                    start_char = gap_start
            # 2. Если текст в дырке сам является валидной сущностью
            elif gap_text:
                for etype in ["BRAND", "VOLUME", "PERCENT", "TYPE"]:
                    if validate_entity(etype, gap_text):
                        merged.append((gap_start, gap_end, f"B-{etype}"))
                        break
                else:
                    # 3. Иначе дыра → "O"
                    merged.append((gap_start, gap_end, "O"))

        # Обрабатываем текущий токен
        if label == "O":
            merged.append((start_char, end_char, "O"))
            prev_label = None
            prev_type = None
            prev_end = end_char
            continue

        current_type = label.split("-", 1)[1]

        # Склейка непрерывных сущностей
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

    # Проверяем хвост после последней сущности
    if prev_end < len(text) - 1:
        gap_start = prev_end + 1
        gap_text = text[gap_start:].strip()
        if gap_text:
            # если остаток — известная сущность
            added = False
            for etype in ["BRAND", "VOLUME", "PERCENT", "TYPE"]:
                if validate_entity(etype, gap_text):
                    merged.append((gap_start, len(text) - 1, f"B-{etype}"))
                    added = True
                    break
            if not added:
                merged.append((gap_start, len(text) - 1, "O"))

    return merged

def predict_annotations(
    text: str,
    keep_O: bool = True,
    fix_i_without_b: bool = True,
) -> list:

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
    n_words = max(word_to_token_idxs.keys()) + 1 if word_to_token_idxs else 0

    for word_id in range(n_words):
        token_idxs = word_to_token_idxs.get(word_id, [])
        if not token_idxs:
            continue

        start_char = int(offsets[token_idxs[0]][0].item())
        end_char = int(offsets[token_idxs[-1]][1].item())

        subtok_labels = [id2label[preds_seq[tok_idx]] for tok_idx in token_idxs]
        label = aggregate_labels(subtok_labels)

        # Фиксим "I-" без "B-"
        if label.startswith("I-") and fix_i_without_b:
            typ = label.split("-", 1)[1]
            prev_typ = None
            if prev_word_label and prev_word_label != "O":
                prev_typ = prev_word_label.split("-", 1)[1]
            if prev_word_label is None or prev_word_label == "O" or prev_typ != typ:
                label = "B-" + typ

        if label == "O" and not keep_O:
            prev_word_label = "O"
            continue

        raw_annotations.append((start_char, end_char, label))
        prev_word_label = label

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
    logger.info("Запуск процесса загрузки модели...")

    # Сначала скачиваем файлы
    await loop.run_in_executor(None, download_from_bucket, best_checkpoint)

    # Потом загружаем модель
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


@app.get("/health")
async def health():
    return {"status": "ok... Mb)"}
