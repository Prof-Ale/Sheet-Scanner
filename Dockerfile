FROM python:3.11-slim

WORKDIR /app

# Install MuseScore CLI & OpenCV runtime dependencies for cloud deployment
RUN apt-get update && apt-get install -y --no-install-recommends \
    musescore3 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
ENV PORT=7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
