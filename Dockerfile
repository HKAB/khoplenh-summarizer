FROM nvcr.io/nvidia/pytorch:24.05-py3

RUN apt-get update && apt-get install ffmpeg -y
RUN pip install yt-dlp

# Prevent pyannote-audio install higher version of torch
RUN pip install insanely-fast-whisper openai

# RUN cd /workspace && git clone https://github.com/HKAB/khoplenh-summarizer.git