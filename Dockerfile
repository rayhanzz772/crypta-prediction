FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirement.txt ./

# Install from requirement.txt when provided; otherwise install minimal runtime deps.
RUN set -eux; \
    if [ -s requirement.txt ]; then \
        pip install --no-cache-dir -r requirement.txt; \
    else \
        pip install --no-cache-dir flask numpy joblib scikit-learn; \
    fi

COPY . .

EXPOSE 5001

CMD ["python", "app.py"]
