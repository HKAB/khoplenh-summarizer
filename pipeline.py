import os
from yt_dlp import YoutubeDL
import sys
import subprocess
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import requests
import json
from datetime import date
import time

CACHE_DIR = "/work/cache"
OVERLAP_CHUNK = 10
MAX_NEW_TOKENS = 512
MAX_CHUNK_LENGTH = 2000

SYSTEM_PROMPT = f"""Bạn là một trợ lí đầu tư chứng khoán Tiếng Việt. Nhiệm vụ của bạn là tổng hợp lại thông tin chính trong bài nói, đưa ra các ý chính quan trọng, thông tin về các mã cổ phiếu được nhắc đến.\n"""
SYSTEM_PROMPT += "Không tự ý thêm thông tin, không đưa ra các thông tin không hữu ích, không quá ngắn.\n"
SYSTEM_PROMPT += "TAKE A DEEP BREATH!\n"

# yt-dlp --extract-audio --audio-format wav --postprocessor-args "-ar 16000" --download-archive downloaded.log --max-filesize 400.0M --max-downloads 10000 --output "%(uploader)s/%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s" -i https://www.youtube.com/@VTVSHOWS/playlists
ydl_opts = {
    "_warnings": [
        "Post-Processor arguments given without specifying name. The "
        "arguments will be given to all post-processors"
    ],
    "download_archive": "downloaded.log",
    "extract_flat": "discard_in_playlist",
    "final_ext": "wav",
    "format": "bestaudio/best",
    "fragment_retries": 10,
    "ignoreerrors": True,
    "max_downloads": 10000,
    "max_filesize": 419430400,
    "paths": {"home": "./temp"},
    "outtmpl": {"default": "output.%(ext)s"},
    "postprocessor_args": {"default": ["-ar", "16000"], "sponskrub": []},
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "nopostoverwrites": False,
            "preferredcodec": "wav",
            "preferredquality": "5",
        },
        {"key": "FFmpegConcat", "only_multi_video": True, "when": "playlist"},
    ],
    "retries": 10,
}


def check_vllm_available():
    try:
        response = requests.get("http://0.0.0.0:8000/v2/health/ready")
        if response.status_code == 200:
            return True
    except Exception as e:
        print("Triton server is not available yet")
        print(e)
    return False


def vllm_generate(text_input, temperature=0, max_tokens=1024, stream=False):
    url = "http://0.0.0.0:8000/v2/models/vllm-model/generate"
    data = {
        "text_input": text_input,
        "parameters": {
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    }

    response = requests.post(url, data=json.dumps(data))
    return response.json()


def download_audio(url):
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":
    if download_audio(sys.argv[1]):
        print("Downloaded audio successfully!")
    else:
        print("ERROR! Downloaded audio failed")
        sys.exit(0)

    if not os.path.exists("./temp/output.json"):
        # insanely-fast-whisper --file-name /work/khoplenh-summarizer/VTVMoney/NA/output.mp3 --device-id 0 --model-name openai/whisper-large-v3 --language vi
        subprocess.run(
            [
                "insanely-fast-whisper",
                "--file-name",
                "./temp/output.wav",
                "--device-id",
                "0",
                "--model-name",
                "openai/whisper-large-v3",
                "--language",
                "vi",
                "--transcript-path",
                "./temp/output.json",
            ]
        )

        print("Transcribed!")
    else:
        print("Transcribe file existed!")

    whisper_output = json.load(open("./temp/output.json", "r"))

    if not os.path.exists("./temp/output.txt"):
        # Run tritonserver command
        if not check_vllm_available():
            triton_env = os.environ.copy()
            triton_env["CUDA_VISIBLE_DEVICES"] = "0"
            proc = subprocess.Popen(
                ["tritonserver", "--model-store", "./model_repository"],
                env=triton_env,
            )
        else:
            print("Tritonserver is already running")

        # Wait for tritonserver to start
        waited_time = 0
        while True:
            if check_vllm_available():
                break
            else:
                waited_time += 1
                print("Waiting for tritonserver to start...")
                if waited_time * 4 > 60:
                    print("ERROR: Timeout waiting for tritonserver")
                    sys.exit(0)
                time.sleep(4)

        tokenizer = AutoTokenizer.from_pretrained(
            "Viet-Mistral/Vistral-7B-Chat", cache_dir=CACHE_DIR
        )

        all_trans_filtered = []
        for chunk in whisper_output["chunks"]:
            if "Hãy subscribe cho kênh" in chunk["text"]:
                continue
            all_trans_filtered.append(chunk)

        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        content = [chunk["text"] + "\n" for chunk in all_trans_filtered]

        chunk_content = []
        current_chunk = ""

        sent_idx = 0
        while sent_idx < len(content):

            temp_conversation = conversation + [
                {"role": "user", "content": current_chunk + content[sent_idx]}
            ]
            input_ids = tokenizer.apply_chat_template(
                temp_conversation, return_tensors="pt"
            )
            # print(sent_idx, input_ids.shape[1])
            if input_ids.shape[1] > MAX_CHUNK_LENGTH:
                human = f"""
        {current_chunk}
        ====================

        Tổng hợp thông tin về thị trường:
            (Các thông tin)
        """
                conversation.append({"role": "user", "content": human})
                input_ids = tokenizer.apply_chat_template(
                    conversation, return_tensors="pt"
                )
                chunk_content.append(input_ids)

                if sent_idx == len(content) - 1:
                    break

                sent_idx -= OVERLAP_CHUNK
                current_chunk = ""

                conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
                # slide back OVERLAP_CHUNK sentences, overlap OVERLAP_CHUNK sentences
            else:
                current_chunk = current_chunk + content[sent_idx]
                sent_idx += 1

        all_summaries = []
        for chunk in tqdm(chunk_content[:]):
            text_input = tokenizer.batch_decode(chunk, skip_special_tokens=True)[
                0
            ].strip()
            max_new_tokens = MAX_NEW_TOKENS + chunk.shape[1]
            try:
                response = vllm_generate(
                    text_input, temperature=0, max_tokens=max_new_tokens, stream=False
                )
            except Exception as e:
                print("ERROR: VLLM can't summarize the text")
                print(e)
                sys.exit(0)

            all_summaries.append(response["text_output"][len(text_input) :].strip())

        all_text_outputs = "\n\n".join(all_summaries)
        with open("./temp/output.txt", "w+") as f:
            f.write(all_text_outputs)
        if proc:
            proc.terminate()
    else:
        all_text_outputs = open("./temp/output.txt", "r").read()

    # CUSTOM POST-PROCESSING
    stock_in_news = set()

    news_bullets = all_text_outputs.split("\n")
    for news_bullet in news_bullets:
        words = news_bullet.split(" ")
        for word in words:
            # check if word is full uppercase
            if word.isupper() and len(word) > 1 or word == "VN-Index":
                stock_in_news.add(word.replace(",", "").replace(".", ""))

    stock_time_refer = {}
    for stock in stock_in_news:
        for chunk_info in whisper_output["chunks"]:
            start, end = chunk_info["timestamp"]
            if stock in chunk_info["text"]:
                if stock not in stock_time_refer:
                    stock_time_refer[stock] = [(start, end)]
                else:
                    stock_time_refer[stock].append((start, end))

    html_content = "---\n"
    html_content += f"layout: stock_post\ntitle: Khớp Lệnh {date.today().strftime('%d/%m/%Y')}\nexcerpt: summarization of 'Khop Lenh today'\ncategories: Invest\n"
    html_content += "---\n\n"

    video_id = sys.argv[1].split("/")[-1]
    html_content += f'<iframe id="player" src="https://www.youtube.com/embed/{video_id}?enablejsapi=1" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>'
    html_content += "\n\n"

    html_content += "<table><tr><th>Stock</th><th>Time mention</th></tr>"
    for stock, times in stock_time_refer.items():
        merge_times = []

        for start, end in times:
            if len(merge_times) == 0:
                merge_times.append([start, end])
            else:
                if start - merge_times[-1][1] < 100:
                    merge_times[-1][1] = end
                else:
                    merge_times.append([start, end])

        # html_content += f"{stock}"
        html_content += f"<tr><td scope='row'>{stock}</td><td>"
        for i, (start, end) in enumerate(merge_times):
            # time = f"{int(start//60)}:{int(start%60)} - {int(end//60)}:{int(end%60)}"
            # create time in format 00:00 - 00:00, if the second from start is less than 10, add 0 before it
            time = f"{int(start//60)}:{int(start%60):02d} - {int(end//60)}:{int(end%60):02d}"
            html_content += f"<a onclick='go_to({start})'>[{time}] </a>"
        html_content += "</td></tr>"
        # += f"<a onclick='go_to({start})'>[{i}] </a>"
    html_content += "</table>"

    html_content += "\n\n"
    html_content += all_text_outputs

    with open(f'./temp/{date.today().strftime("%d%m%Y")}.md', "w") as f:
        f.write(html_content)
