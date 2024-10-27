# https://twitter.com/clcoding/status/1695667204531064905

from wordcloud import WordCloud
import matplotlib.pyplot as plt 

# Open the text file
FILE = "/home/jeff/Downloads/scratch/instguid.git/hlmGPT/data/books/hongloumeng_xiaozhuben__caoxueqin.txt"
with open(FILE, 'r', encoding="utf-8") as f:
    text = f.read()

# Generate the word cloud
wordcloud = WordCloud(background_color="black", font_path="./SimHei.ttf", width=800, height=400)
wordcloud.generate(text)

# Display the word cloud
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
# pls.savefig("test.png")
plt.show()
