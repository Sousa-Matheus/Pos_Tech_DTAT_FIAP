# Deploy Modelo de Séries Temporais com StreamLit
# Importando as bibliotecas
import streamlit as st
from datetime import datetime, timedelta


hoje = datetime.today()

amanha = datetime.today() + timedelta(days=1)


# Configura layout wide
st.set_page_config(page_title="Previsão do Petróleo", layout="wide")

# Título centralizado (opcional)
st.title("Previsão do valor do barril de petróleo")

# Cria 3 colunas: esquerda, meio, direita
col1, col2, col3 = st.columns([1, 2, 1])

# Exibe a data apenas na primeira coluna (à esquerda)
with col1:
    st.text_area('Data de hoje:', value=hoje.strftime("%d/%m/%Y"), height=68)


st.slider("Escolha o número de dias para previsão", 1, 30, 15)
st.text_area('Previsão para Amanhã:', value=amanha.strftime("%d/%m/%Y"), height=68)
st.text_area('Previsão do valor do barril de petróleo para amanhã:', value="R$ 0,00", height=68)

st.text_area('Gráfico de Previsão:', value="Gráfico", height=68)
st.line_chart([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])



