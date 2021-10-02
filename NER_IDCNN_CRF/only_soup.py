# encoding=utf8

def evaluate_line(texts):
    for html_text in texts:
        text = extract_plaintext(html_text)
        # result = model.evaluate_line(sess, input_from_line(text, char_to_id), id_to_tag)
        # ans.append(result['entities'])
        

from bs4 import BeautifulSoup

def extract_plaintext(html_text):
    # html_text = html_text.encode('UTF-8')
    # print(html_text)
    soup = BeautifulSoup(html_text, 'html.parser')
    # text = soup.get_text()
    text = ''
    for i in soup.stripped_strings:
        tmp = ''.join(i.split())
        if tmp.startswith(('审判长', '审判员', '代理审判员', '人民陪审员', '书记员', '执行员')):
            break
        # if i.startswith(('审判长', '审　判　长', '审 判 长', '审　 判　 长', '审  判  长', '审   判    长')):
        #     break
        # if i.startswith(('审判员', '审　判　员', '审 判 员', '审　 判　 员', '审  判  员', '审   判    员')):
        #     break
            
        text += i
    return text


if __name__ == "__main__":
    # with open('./out','r') as f:
    #     html_text = f.read()
    # print(html_text)
    # soup = BeautifulSoup(html_text, 'html.parser')
    # for i in soup.stripped_strings:
    #     print(i, '!')
    # exit()
    import torch
    texts = torch.load('../datasets/pre/content.pt')
    texts = evaluate_line(texts)
    # torch.save(texts,'../datasets/feature/entity.pt')


