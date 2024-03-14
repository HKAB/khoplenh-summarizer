from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import pickle
import os
import requests
import json

CACHE_DIR = './cache'
OVERLAP_CHUNK = 10
MAX_NEW_TOKENS = 512

def vllm_generate(text_input, temperature=0, max_tokens=1024, stream=False):
    url = "http://localhost:9000/v2/models/vllm_model/generate"
    data = {
        "text_input": text_input,
        "parameters": {
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    }

    response = requests.post(url, data=json.dumps(data))
    return response.json()

if __name__ == "__main__":
    
    if not os.path.exists('./whisper_seg.pkl'):
        with open('./whisper_seg.pkl', 'rb') as f:
            trans_data = pickle.load(f)
    else:
        print("ERROR: Transcription file not found")
        sys.exit(1)

    system_prompt = f"""Bạn là một trợ lí đầu tư chứng khoán Tiếng Việt. Nhiệm vụ của bạn là tổng hợp lại thông tin chính trong bài nói, đưa ra các ý chính quan trọng, thông tin về các mã cổ phiếu được nhắc đến.\n"""
    system_prompt += "Không tự ý thêm thông tin, không đưa ra các thông tin không hữu ích, không quá ngắn.\n"
    system_prompt += "TAKE A DEEP BREATH!\n"

    tokenizer = AutoTokenizer.from_pretrained('Viet-Mistral/Vistral-7B-Chat', cache_dir=CACHE_DIR)

    conversation = [{"role": "system", "content": system_prompt }]
    content = [s[2] + "\n" for s in trans_data]

    chunk_content = []
    current_chunk = ""

    
    sent_idx = 0
    while sent_idx < len(content):
        
        temp_conversation = conversation + [{"role": "user", "content": current_chunk + content[sent_idx] }]
        input_ids = tokenizer.apply_chat_template(temp_conversation, return_tensors="pt")
        # print(sent_idx, input_ids.shape[1])
        if input_ids.shape[1] > 2000:
            human = f"""
    {current_chunk}
    ====================

    Tổng hợp thông tin về thị trường:
        (Các thông tin)
    """
            conversation.append({"role": "user", "content": human })
            input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
            chunk_content.append(input_ids)

            if sent_idx == len(content) - 1:
                break
        
            sent_idx -= OVERLAP_CHUNK
            current_chunk = ""

            conversation = [{"role": "system", "content": system_prompt }]
            # slide back OVERLAP_CHUNK sentences, overlap OVERLAP_CHUNK sentences
        else:
            current_chunk = current_chunk + content[sent_idx]
            sent_idx += 1

    all_summaries = []
    for chunk in tqdm(chunk_content[:]):
        text_input = tokenizer.batch_decode(chunk, skip_special_tokens=True)[0].strip()
        max_new_tokens = MAX_NEW_TOKENS + chunk.shape[1]
        try:
            response = vllm_generate(text_input, temperature=0, max_tokens=max_new_tokens, stream=False)
        except Exception as e:
            print("ERROR: VLLM can't summarize the text")
            print(e)
            sys.exit(1)
        
        all_summaries.append(response['text_output'][len(text_input):].strip())


    all_text_outputs = "\n\n".join(all_summaries)

    # CUSTOM POST-PROCESSING
    stock_in_news = set()

    news_bullets = all_text_outputs.split("\n")
    for news_bullet in news_bullets:
        words = news_bullet.split(" ")
        for word in words:
            # check if word is full uppercase
            if word.isupper() and len(word) > 1 or word == "VN-Index":
                stock_in_news.add(word.replace(",", '').replace(".", ""))

    stock_time_refer = {}
    for stock in stock_in_news:
        for start, end, trans in trans_data:
            if stock in trans:
                if stock not in stock_time_refer:
                    stock_time_refer[stock] = [(start, end)]
                else:
                    stock_time_refer[stock].append((start, end))

    # html_content = ""
    html_content = '<table><tr><th>Stock</th><th>Time mention</th></tr>'
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

    with open('./summary.html', 'w') as f:
        f.write(html_content)