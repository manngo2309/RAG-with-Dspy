import streamlit as st
import openai
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os 
import chromadb
from chromadb.utils import embedding_functions
import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
import time
from utils import update_db

##
# from rag_model import *
from rag_model_with_assert import *
import rag_model_with_assert
import utils


load_dotenv() 
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# embedding model
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                model_name="text-embedding-ada-002", 
                api_key = OPENAI_API_KEY)

retriever_q = ChromadbRM(
    'vcb_qa7',
    persist_directory='./data',
    embedding_function=openai_ef,
    k=5
)
retriever_f = ChromadbRM(
    'vcb_f7',
    persist_directory='./data',
    embedding_function=openai_ef,
    k=5
)

### load lm
llm = dspy.OpenAI(model='gpt-3.5-turbo',max_tokens=1000, temperature = 0.01)
# llm = dspy.OpenAI(model='gpt-4o',max_tokens=300, temperature = 0.01)

dspy.settings.configure(lm=llm)
dspy.settings.bypass_suggest = False

### init rag model
# citation_model = LongFormQAWithAssertions(retriever_q=retriever_q,retriever_f = retriever_f).activate_assertions(max_backtracks=3)  # uncompiled (i.e., zero-shot) program
without_citation_model = SimplifiedBaleen(retriever_q=retriever_q,retriever_f = retriever_f)  # uncompiled (i.e., zero-shot) program
# longformqa_with_assertions = LongFormQAWithAssertions().activate_assertions(max_backtracks=3)
# citation_model = assert_transform_module(LongFormQAWithAssertions(retriever_q=retriever_q,retriever_f = retriever_f).map_named_predictors(Retry), backtrack_handler,max_backtracks=3) 


citation_model = LongFormQAWithAssertions(retriever_q=retriever_q,retriever_f = retriever_f).activate_assertions(max_backtracks=3)  # uncompiled (i.e., zero-shot) program
citation_model = assert_transform_module(LongFormQAWithAssertions(retriever_q=retriever_q,retriever_f = retriever_f).map_named_predictors(Retry), backtrack_handler) 
generate_query_from_chat = dspy.Predict(GenerateSearchQueryFromChatHist)


#### clear cache
st.cache_data.clear()
st.cache_resource.clear()
####
# Initialize the chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'chat_hist' not in st.session_state:
    st.session_state['chat_hist'] = []
# Function to display chat messages
def display_messages():
    for message in st.session_state['messages']:
        if message['is_user']:
            st.markdown(f"<div style='text-align: right; background-color: #dcf8c6; padding: 5px 10px; border-radius: 10px; margin: 5px;'>{message['text']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left; background-color: #f1f0f0; padding: 5px 10px; border-radius: 10px; margin: 5px;'>{message['text']}</div>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    body {
        color: black;
    }
    .stMarkdown {
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("RAG with VCB FAQ")

# Display chat messages
display_messages()

# Input box for user to type messages
user_input = st.text_input("You:", key="input_text")
# Handle user input

if st.button("Send", key="send_button"):
    with st.spinner("Xin chờ..."):
        if user_input:
            # Add user message to chat history
            st.session_state['messages'].append({"text": user_input, "is_user": True})
            st.session_state['chat_hist'].append("User: " + user_input)
            ### paraphrase question 
            # if len(st.session_state['chat_hist'])>0:
            #     print("old",user_input)
            #     user_input = generate_query_from_chat(context = "\n".join(st.session_state['chat_hist']), question = user_input).query
            #     print(user_input)
            try:
                    ### try to get citation <-> avoid hallucination
                pred = citation_model(user_input,"\n".join(st.session_state['chat_hist'][-3:-1]))
                bot_response = pred.paragraph.replace("\n","<br>")
                st.session_state['chat_hist'].append("Bot: " + bot_response)

                citation = "\n\n************* CITATION ****************\n"
                for e in extract_text_by_citation(pred.paragraph):
                    # try:
                        citation += ("**************\n" + f"[{int(e)+1}] " + pred.context[int(e)] +"\n****************\n")
                    # except:
                    #     print( pred.context.keys() )
                bot_response += citation
                bot_response = bot_response.replace("\n","<br>")
                
            except:
                ## multi-hop without citation
                print("******************")
                bot_response =  without_citation_model(user_input,"\n".join(st.session_state['chat_hist'][-3:-1])).answer.replace("\n","<br>")
                st.session_state['chat_hist'].append("Bot: " + bot_response)

            # Add bot response to chat history
            st.session_state['messages'].append({"text": bot_response, "is_user": False})
 

    # Rerun to display updated messages
    st.experimental_rerun()



if st.button("Update DB (only when we need to crawl and update db)", key="updatedb_button"):
    update_db()
    retriever_q = ChromadbRM(
    'vcb_qa7',
    persist_directory='./data',
    embedding_function=openai_ef,
    k=5
)
    retriever_f = ChromadbRM(
        'vcb_f7',
        persist_directory='./data',
        embedding_function=openai_ef,
        k=5
    )
