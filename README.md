
# VTV Khop Lenh AI summarizer 📈

| All I need is money to become rich. - Anonymous

![Stonk](./images/intro.jpg)

Transcribe and summarize Khop Lenh's video using Whisper and LLM


## Run Locally (So easy now!) 🚀

1) Setup model_repository folder for vLLM
- Setup model_repository as in [vllm_backend](https://github.com/triton-inference-server/vllm_backend/tree/main/samples/model_repository)

```json
// model.json
{
    "model":"Viet-Mistral/Vistral-7B-Chat",
    "disable_log_requests": "true",
    "gpu_memory_utilization": 1.0,
    "dtype": "bfloat16",
    "download_dir": "/workspace/cache_dir"
}
```

2) Create only 1 docker by running:

```sh
bash run.sh
```

3) Wait util the bash show up, then run:

```sh
python3 pipeline.py <YOUTUBE_LINK>
```

4) Result is written in markdown file in `temp` folder!

## To do 📝

- [x] Automate all this. 
- [ ] Add diarization in Whisper pipeline all this. 


## Feedback 📢

- Anonymous [Nguyen Manh Khang](https://github.com/nguyenbim/): OMG I love this, Khop Lenh summarizer help me to invest much better. My portfolio is up 9669% in just 2 weeks. Thank you so much!

- Anonymous Broker: AI freak me out. I'm out of job now. Please stop developing this tool. I need to feed my family. SIRR...

## Thanks 🙏
- [Vistral](https://huggingface.co/Viet-Mistral/Vistral-7B-Chat)
- [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper)