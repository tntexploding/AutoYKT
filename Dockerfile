FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AUTOYKT_SERVER_CONFIG=/config/server.yaml

WORKDIR /app
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/server-requirements.txt \
    && pip install --no-cache-dir 'setuptools>=68' wheel -r /tmp/server-requirements.txt

COPY README.md LICENSE ./
COPY src ./src
COPY third_party ./third_party
RUN pip install --no-cache-dir --no-deps --no-build-isolation . \
    && useradd --uid 10001 --create-home autoykt \
    && mkdir -p /data /config \
    && chown autoykt:autoykt /data

USER autoykt
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ["python", "-m", "autoykt.server", "health"]
ENTRYPOINT ["python", "-m", "autoykt.server"]
CMD ["run"]
