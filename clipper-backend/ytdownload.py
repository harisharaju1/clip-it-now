from pytubefix import YouTube
from pytubefix.cli import on_progress

# url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
url2 = "https://www.youtube.com/watch?v=IsScahXkvMk"

yt = YouTube(url2, on_progress_callback=on_progress)

print(yt.title)
print(yt.views)
print(yt.length)
print(yt.description)
print(yt.author)

ys = yt.streams.get_highest_resolution()
ys.download()
