
# VTV Khop Lenh AI summarizer 📈

| All I need is money to become rich. - Anonymous

![Stonk](./images/intro.jpg)

Transcribe and summarization of Khop Lenh using Whisper and LLM


## Run Locally 🚀

1) Create main docker
- I'm using `pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel` (make sure `--net host` for ease to receive llm output from vLLM, or mapping port which is exposed by vLLM docker)
- Install Whisper by follow the setup in [m-bain/whisperX](https://github.com/m-bain/whisperX?tab=readme-ov-file#setup-%EF%B8%8F)
- Install https://pypi.org/project/yt-dlp/

2) Create a second docker for vLLM
- Setup model_repository as in [vllm_backend](https://github.com/triton-inference-server/vllm_backend/tree/main/samples/model_repository)

```json
// model.json
{
    "model":"Viet-Mistral/Vistral-7B-Chat",
    "disable_log_requests": "true",
    "gpu_memory_utilization": 0.5,
    "dtype": "bfloat16",
    "download_dir": "/workspace/cache_dir"
}
```

```sh
docker run --gpus all -it -p 9000-9002:8000-8002 --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v path/to/workspace:/workspace -w /workspace nvcr.io/nvidia/tritonserver:23.10-vllm-python-py3 tritonserver --model-store ./model_repository
```

Then run:

```sh
python3 download_wav.py <youtube_url>
python3 transcribe.py <wav_path>
python3 summarize.py
```


## To do 📝

- [ ] Automate all this. 


## Feedback 📢

- Anonymous [Nguyen Manh Khang](https://github.com/nguyenbim/): OMG I love this, Khop Lenh summarizer help me to invest much better. My portfolio is up 9669% in just 2 weeks. Thank you so much!

- Anonymous Broker: AI freak me out. I'm out of job now. Please stop developing this tool. I need to feed my family. SIRR...

