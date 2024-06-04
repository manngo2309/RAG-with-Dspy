import regex as re

def extract_text_by_citation(paragraph):
    citation_regex = re.compile(r'(.*?)(\[\d+\]\.)', re.DOTALL)
    parts_with_citation = citation_regex.findall(paragraph)
    citation_dict = {}
    for part, citation in parts_with_citation:
        part = part.strip()
        citation_num = re.search(r'\[(\d+)\]\.', citation).group(1)
        citation_dict.setdefault(str(int(citation_num) - 1), []).append(part)
    return citation_dict



def has_citations(paragraph):
    return bool(re.search(r'\[\d+\]\.', paragraph))

def citations_check(paragraph):
    print("has_citations(paragraph)",has_citations(paragraph))
    return has_citations(paragraph) #and correct_citation_format(paragraph)



 #Update chromadb
def update_db():
    from dotenv import load_dotenv
    import os 
    import chromadb
    from chromadb.utils import embedding_functions
    import numpy as np
    import pandas as pd

    load_dotenv() 
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    model_name="text-embedding-ada-002", 
                    api_key = OPENAI_API_KEY)


    np.random.seed(42) 
    data_cn = pd.read_csv("data/data_cn.csv")
    # data_cn = data_cn[['question','answer']]#.sample(10)
    data_cn = data_cn.drop_duplicates()#.shape

    data_cn['id'] = [str(s) for s in data_cn.index.values.tolist()]
    data_cn['full'] = data_cn.apply(lambda x: "Câu hỏi: " + x['question'] + "\n" + "Câu trả lời: " + x['answer'] +"\n###" if x['question']!=x['answer'] else x['question'],axis = 1)

    #Update chromadb

    chroma_client = chromadb.PersistentClient(path="./data")

    collection = chroma_client.get_or_create_collection(name="vcb_qa7", embedding_function=openai_ef)
    collection2 = chroma_client.get_or_create_collection(name="vcb_f7", embedding_function=openai_ef)

    collection.add(
        documents = data_cn['question'].tolist(),
        metadatas = [{"full":s} for s in data_cn['full'].tolist()],
        ids = data_cn['id'].tolist()
    )   
    collection2.add(
        documents = data_cn['full'].tolist(),
        metadatas = [{"full":s} for s in data_cn['full'].tolist()],
        ids = data_cn['id'].tolist()
    )     