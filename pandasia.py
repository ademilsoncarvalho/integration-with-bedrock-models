import os
import pandas as pd
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import streamlit as st

load_dotenv()
open_ia_token = os.getenv("OPENAI_API_KEY")

llm = OpenAI(api_token=open_ia_token)

# Title of the web app
st.title('Analise de dados com IA')

# Upload a file
uploaded_file = st.file_uploader("Escolha oa arquivo CSV para analisar:")

# If a file is uploaded
if uploaded_file is not None:
    print(uploaded_file)
    df = pd.read_csv(uploaded_file)
    data_frame = SmartDataframe(df, config={"llm": llm})
    st.write(df.head(5))

    prompt = st.text_input('O que precisa saber sobre os dados?')
    if st.button('Perguntar'):
        if prompt:
            with st.spinner('Aguarde...'):
                response = data_frame.chat(prompt)
                st.write(response)
        else:
            st.warning('Por favor, insira uma pergunta.')

