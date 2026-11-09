FROM python:3.12.8-slim
ARG WHEEL=autofusedebugger-0.1.2-py3-none-any.whl
COPY ${WHEEL} /tmp/package.whl
RUN mv /tmp/package.whl /tmp/${WHEEL} && pip install --no-cache-dir --no-deps /tmp/${WHEEL} && rm /tmp/${WHEEL} && useradd -u 10001 -m fuzz
USER 10001
EXPOSE 8090
ENTRYPOINT ["autofusedbg", "fuzz", "serve", "/data/results.sqlite", "--port", "8090"]
