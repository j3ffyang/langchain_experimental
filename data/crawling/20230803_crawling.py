import requests
from bs4 import BeautifulSoup

# url = 'http://m.sbkk8.com/gudai/sidawenxuemingzhu/caiyijiangjieduhongloumeng/'
url = 'http://www.guoxue123.com/hongxue/0001/lxwjmhl/index.htm'

headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
    'Cache-Control': 'max-age=0',
    'Cookie': 'UM_distinctid=189b956fb5e17-05ec2d984c523c-14462c6c-1fa400-189b956fb604ff; CNZZDATA1281278841=1371756866-1691030548-%7C1691030548; Hm_lvt_439e35470defdfe66d70b6886ac84ba3=1691031641; Hm_lpvt_439e35470defdfe66d70b6886ac84ba3=1691031641',
    'Host': 'm.sbkk8.com',
    'If-Modified-Since': 'Thu, 11 Aug 2022 12:49:31 GMT',
    'If-None-Match': 'W/"62f4fadb-2972"',
    'Proxy-Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
}

response = requests.post(url, headers=headers, allow_redirects=True)
soup = BeautifulSoup(response.text, 'html.parser')

print(response)

# with open('links.txt', 'w') as f:
#     for line in soup.findall('a'):
#         f.write(link.get('href') + '\n')
