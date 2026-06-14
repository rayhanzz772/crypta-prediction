FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5010

WORKDIR /app

COPY requirement.txt ./

RUN pip install --no-cache-dir -r requirement.txt

COPY . .

EXPOSE 5010

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/' % __import__('os').environ.get('PORT','5010'), timeout=3)"

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5010} --workers ${GUNICORN_WORKERS:-1} --threads ${GUNICORN_THREADS:-1} --timeout ${GUNICORN_TIMEOUT:-120} app:app"]
