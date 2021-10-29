from bs4 import BeautifulSoup
import jionlp as jio

def getText(e):

    soup = BeautifulSoup(e, 'html.parser')
    text = soup.get_text()

    return text

def extract_plaintext(html_text):
    # 预处理！
    soup = BeautifulSoup(html_text, 'html.parser')
    # text = soup.get_text()
    text = []
    for i in soup.stripped_strings:
        tmp = ''.join(i.split())
        if tmp.startswith(('审判长', '审判员', '代理审判员', '人民陪审员', '书记员', '执行员')):
            break
        tmp = jio.split_sentence(tmp, criterion='coarse')
        text.extend(tmp)
    # print(text)
    return text


