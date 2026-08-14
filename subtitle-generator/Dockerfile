FROM nvidia/cuda:12.1-runtime-ubuntu22.04

RUN apt update && apt install -y python3.10 python3-pip ffmpeg && \
    pip3 install faster-whisper pyannote.audio anthropic librosa soundfile flask flask-cors deep-translator fugashi unidic-lite

WORKDIR /app
COPY server.py generate_subtitles.py index.html ./

ENV HF_TOKEN=""
EXPOSE 5000

CMD ["python3", "server.py"]
