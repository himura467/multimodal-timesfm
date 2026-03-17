FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
RUN apt update -y && \
  apt install -y wget git && \
  wget -qO- https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
RUN git clone https://github.com/himura467/tsfmx.git && \
  cd tsfmx && \
  ./scripts/clone_time_mmd.sh && \
  uv sync --extra sweep
WORKDIR /tsfmx
ENV PYTHONPATH=/tsfmx
RUN uv run python scripts/split_time_mmd_datasets.py \
  --train-ratio 0.6 \
  --val-ratio 0.2
