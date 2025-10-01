# Используем официальный образ Python
FROM python:3.11-slim

# Рабочая директория в контейнере
WORKDIR /app

# Копируем файлы проекта
COPY . /app

# Обновляем pip и устанавливаем зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi \
        uvicorn[standard] \
        torch \
        transformers \
        torchcrf \
        pandas

# Указываем порт, который будет слушать приложение
EXPOSE 8000

# Команда запуска
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
