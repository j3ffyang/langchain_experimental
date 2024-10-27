import requests
from bs4 import BeautifulSoup

url = 'http://m.sbkk8.com/gudai/sidawenxuemingzhu/caiyijiangjieduhongloumeng/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

with open('crawling_links.txt', 'w') as f:
    for link in soup.find_all('a'):
       f.write(link.get('href') + '\n')

with open('crawling_links.txt', 'r') as f:
    links = f.readlines()

with open('extracted_data.txt', 'w') as f:
    for link in links:
        response = requests.get(link.strip())
        soup = BeautifulSoup(response.text, 'html.parser')
        f.write(soup + '\n')