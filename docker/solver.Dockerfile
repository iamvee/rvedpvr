# Pinned by digest: dolfinx 0.11 (py3.12); update with `docker buildx imagetools inspect dolfinx/dolfinx:stable`.
FROM dolfinx/dolfinx:stable@sha256:628f40dd62c7f37962ccadc5a745f23d52c56b2ac3ad9b9b6a86c567c07aa27b

ENV OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir scipy

WORKDIR /work

CMD ["bash"]
