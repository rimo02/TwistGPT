from bs4 import BeautifulSoup
import requests

def Generate_text():
    url_main = "https://short-edition.com/en/classic/author/o-henry"
    response = requests.get(url_main)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = [a['href']
             for a in soup.find_all('a', class_="savoir-plus txt-classique")]
    final = []
    for link in links:
        link = 'https://short-edition.com' + link
        final.append(link)

    with open("final.txt", "w", encoding="utf-8") as f:
        for link in final:
            resp = requests.get(link)
            soup = BeautifulSoup(resp.content, 'html.parser')
            body = soup.find('div', class_='content')
            f.write(body.text + "\n<|endoftext|> \n")


if __name__ == "__main__":
    Generate_text()
