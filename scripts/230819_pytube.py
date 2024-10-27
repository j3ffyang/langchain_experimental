# https://www.freecodecamp.org/news/python-program-to-download-youtube-videos/

# import pytube
from pytube import YouTube
from pytube.cli import on_progress


def download_video(url):
    """Download"""
    yt = YouTube(url, on_progress_callback=on_progress)
    # yt.streams.first().download()
    yt.streams.get_highest_resolution().download("/home/jeff/")


# if __name__ == '__main__':
#     url = input('Enter the url of the video to download: ')
#     download_video(url)


def main():
    """Main"""
    url = input("Enter the url of the video to download: ")
    download_video(url)


if __name__ == "__main__":
    main()
