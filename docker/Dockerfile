FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
RUN apt update -y && \
  apt install -y wget git && \
  wget -qO- https://astral.sh/uv/install.sh | sh && \
  git clone https://github.com/himura467/multimodal-timesfm.git && \
  cd multimodal-timesfm && \
  ./setup_time_mmd.sh && \
  $HOME/.local/bin/uv sync
WORKDIR /multimodal-timesfm
