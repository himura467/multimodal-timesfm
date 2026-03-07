FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
RUN apt update -y && \
  apt install -y wget git && \
  wget -qO- https://astral.sh/uv/install.sh | sh && \
  export PATH="$HOME/.local/bin:$PATH" && \
  git clone https://github.com/himura467/multimodal-timesfm.git && \
  cd multimodal-timesfm && \
  uv sync --extra sweep && \
  ./scripts/setup_time_mmd.sh
WORKDIR /multimodal-timesfm
