import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
#clean text
def clean_text(text):
    # Remove zero-width spaces (\u200b) and non-breaking spaces (\xa0)
    text = text.replace('\u200b\u200b', '\u200b').replace('\xa0\xa0', '\xa0')
    
    # Alternatively, remove all non-printable or non-standard characters using regex
    # text = re.sub(r'[^\x20-\x7E\u00A0-\uD7FF\uE000-\uFFFD]', '', text)
    
    return text

# Function to extract questions and answers
def extract_faqs(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    faqs = []

    for li in soup.select('ul.ul-faqs > li'):
        question = li.find('b').text.strip()
        answer_div = li.find('div', id=lambda x: x and x.startswith('answer-'))
        if answer_div:
            answer = answer_div.get_text(separator="\n", strip=True)
            faqs.append({'question': clean_text(question), 'answer': clean_text(answer)})

    return faqs

if __name__ == "__main__":
    pattern = r'^\d+\.\s'
    # kh ca nhan
    url_cn = "https://portal.vietcombank.com.vn/FAQs/Pages/hoi-dap-ca-nhan.aspx?devicechannel=default"
    # kh doanh nghiep
    url2_dn = "https://portal.vietcombank.com.vn/FAQs/Pages/hoi-dap-to-chuc.aspx?devicechannel=default"
    # Extract FAQs
    faq_cn = extract_faqs(url_cn)
    faq_dn = extract_faqs(url2_dn)
    data_cn = pd.DataFrame(faq_cn)
    data_dn = pd.DataFrame(faq_dn)
    # clean question
    data_cn['question'] = data_cn['question'].apply(lambda x: re.sub(pattern, '',x) )
    data_dn['question'] = data_dn['question'].apply(lambda x: re.sub(pattern, '',x) )

    data_cn.to_csv("data/data_cn.csv",index=False)
    data_dn.to_csv("data/data_dn.csv",index=False)


