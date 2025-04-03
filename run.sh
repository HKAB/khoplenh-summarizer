docker build -f Dockerfile -t truongnp_whisper_openai:latest .
docker stop khoplenh-summarizer
docker container rm khoplenh-summarizer
docker run --gpus all -it --net=host --name khoplenh-summarizer --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace truongnp_whisper_openai:latest bash