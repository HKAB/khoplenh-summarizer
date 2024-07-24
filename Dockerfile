FROM nvcr.io/nvidia/tritonserver:24.04-vllm-python-py3

RUN apt-get update && apt-get install ffmpeg -y
RUN pip install yt-dlp

# Prevent pyannote-audio install higher version of torch
RUN pip install insanely-fast-whisper torch==2.1.2 torchaudio==2.1.2

RUN cd /workspace && git clone https://github.com/HKAB/khoplenh-summarizer.git