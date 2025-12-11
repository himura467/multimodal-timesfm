FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
RUN apt update -y && \
  apt install -y wget git && \
  wget -qO- https://astral.sh/uv/install.sh | sh && \
  git clone https://github.com/himura467/multimodal-timesfm.git && \
  git submodule update --init && \
  cd multimodal-timesfm && \
  $HOME/.local/bin/uv sync
