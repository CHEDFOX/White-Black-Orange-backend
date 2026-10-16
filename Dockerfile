# Backend image.
#
# Multi-stage so the runtime layer carries no build toolchain.
#
# EXTRA_CA_CERT is an optional build argument naming a PEM file inside the
# build context. Set it when building behind a TLS-terminating proxy, which is
# common in corporate networks and in sandboxed CI: without it pip fails with
# "self-signed certificate in certificate chain" and the error does not say
# why. Leave it unset for an ordinary build.
#
#   docker build --build-arg EXTRA_CA_CERT=corp-ca.crt -t backend .

FROM python:3.12-slim AS builder

ARG EXTRA_CA_CERT=""

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libpq-dev ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy the whole context's certificate directory only when one was named, so an
# ordinary build needs no extra file present.
COPY requirements.txt ${EXTRA_CA_CERT:-requirements.txt} ./

RUN if [ -n "$EXTRA_CA_CERT" ]; then \
      cp "$(basename "$EXTRA_CA_CERT")" /usr/local/share/ca-certificates/extra-ca.crt \
      && update-ca-certificates \
      && echo "trusting the supplied CA bundle for this build"; \
    fi \
 && python -m venv /opt/venv \
 && /opt/venv/bin/pip install -r requirements.txt


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
 && apt-get install -y --no-install-recommends libpq5 curl \
 && rm -rf /var/lib/apt/lists/*

# Run unprivileged: nothing here needs root, and a compromised provider
# response should not reach a root process.
RUN useradd --create-home --shell /bin/bash app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app alembic.ini pyproject.toml ./
COPY --chown=app:app alembic ./alembic
COPY --chown=app:app app ./app
COPY --chown=app:app scripts ./scripts

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
