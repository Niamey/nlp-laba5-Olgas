FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

ENV CACHE_DIR=/data/cache
RUN mkdir -p /data/cache

VOLUME ["/data/cache"]

CMD ["python", "server.py"]
