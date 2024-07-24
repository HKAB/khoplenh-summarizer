docker build -f Dockerfile -t tritonserver-vllm-whisper:latest .
docker stop khoplenh-summarizer
docker container rm khoplenh-summarizer
docker run --gpus all -it --net=host --name khoplenh-summarizer --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace tritonserver-vllm-whisper:latest bash