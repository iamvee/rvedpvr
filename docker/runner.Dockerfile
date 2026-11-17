FROM docker:cli AS dockercli
FROM python:3.11-slim
COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker
ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir pyyaml
WORKDIR /work
CMD ["python", "run.py", "--help"]
