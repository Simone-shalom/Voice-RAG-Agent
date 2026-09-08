FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# $PORT is set by PaaS platforms (Railway, Render); defaults to 8000 for
# plain `docker run`. Local dev overrides this whole command in
# docker-compose.yml to add --reload — this CMD is the production path.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
