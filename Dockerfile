FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# PyTorch CPU 运行依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 非特权运行用户
RUN groupadd --system app && useradd --system --gid app --home /app app

# 应用代码；.dockerignore 排除密钥、虚拟环境和持久化数据
COPY --chown=app:app . .

# 创建持久化目录
RUN mkdir -p /app/chroma_data /app/data && chown -R app:app /app

USER app

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os, urllib.request; port = os.getenv('PORT', '8010'); urllib.request.urlopen(f'http://127.0.0.1:{port}/live', timeout=3)"

CMD ["sh", "-c", "exec uvicorn main:app --host \"${HOST:-0.0.0.0}\" --port \"${PORT:-8010}\""]
