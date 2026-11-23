FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PORT=7860 \
    HOME=/home/user

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

COPY --chown=user:user . /app

RUN sed -i 's/\r$//' /app/start_hf.sh \
    && chmod +x /app/start_hf.sh \
    && mkdir -p /app/player_photos/avatars_webp /app/sample_data/pro_analytics /home/user/.streamlit \
    && chown -R user:user /app /home/user

USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7860/_stcore/health || exit 1

CMD ["/app/start_hf.sh"]
