import os
from yt_dlp import YoutubeDL

def download_video(url, output_path="video"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    VIDEO_URL = "https://www.youtube.com/watch?v=6_vQiDlZ10Y"
    download_video(VIDEO_URL)
