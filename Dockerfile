# Базовый образ Python 3.10
FROM python:3.10-slim

# Рабочая директория в контейнере
WORKDIR /app

# Копируем файлы проекта
COPY . /app

# Устанавливаем зависимости для сборки и необходимые библиотеки
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    git \
 && rm -rf /var/lib/apt/lists/*

# Обновляем pip и ставим зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi \
        uvicorn[standard] \
        torch \
        transformers \
        pandas && \
    pip install --no-cache-dir git+https://github.com/kmkurn/pytorch-crf.git

# Указываем порт
EXPOSE 8000

# Команда запуска
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
