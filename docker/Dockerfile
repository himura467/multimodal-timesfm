FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
RUN apt update -y && \
  apt install -y wget git && \
  wget -qO- https://astral.sh/uv/install.sh | sh && \
  export PATH="$HOME/.local/bin:$PATH" && \
  git clone https://github.com/himura467/multimodal-timesfm.git && \
  cd multimodal-timesfm && \
  ./scripts/clone_time_mmd.sh && \
  uv sync --extra sweep
WORKDIR /multimodal-timesfm
ENV PYTHONPATH=/multimodal-timesfm
RUN uv run scripts/split_time_mmd_datasets.py \
  --train-ratio 0.8 \
  --val-ratio 0.1
