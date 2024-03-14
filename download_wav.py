import os
from yt_dlp import YoutubeDL
import sys

# yt-dlp --extract-audio --audio-format wav --postprocessor-args "-ar 16000" --download-archive downloaded.log --max-filesize 400.0M --max-downloads 10000 --output "%(uploader)s/%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s" -i https://www.youtube.com/@VTVSHOWS/playlists
ydl_opts = {'_warnings': ['Post-Processor arguments given without specifying name. The '
               'arguments will be given to all post-processors'],
 'download_archive': 'downloaded.log',
 'extract_flat': 'discard_in_playlist',
 'final_ext': 'wav',
 'format': 'bestaudio/best',
 'fragment_retries': 10,
 'ignoreerrors': True,
 'max_downloads': 10000,
 'max_filesize': 419430400,
 'outtmpl': {'default': '%(uploader)s/%(playlist)s/%(playlist_index)s - '
                        '%(title)s.%(ext)s'},
 'postprocessor_args': {'default': ['-ar', '16000'], 'sponskrub': []},
 'postprocessors': [{'key': 'FFmpegExtractAudio',
                     'nopostoverwrites': False,
                     'preferredcodec': 'wav',
                     'preferredquality': '5'},
                    {'key': 'FFmpegConcat',
                     'only_multi_video': True,
                     'when': 'playlist'}],
 'retries': 10}

# URLS = ['https://youtu.be/JmpkOuBXmkI']
# with YoutubeDL(ydl_opts) as ydl:
#     ydl.download(URLS)

# write main method, receive url, download audio, then print 'OK'

def download_audio(url):
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        print(e)
        return False

if __name__ == '__main__':
    if download_audio(sys.argv[1]):
        print('OK')
    else:
        print('ERROR')