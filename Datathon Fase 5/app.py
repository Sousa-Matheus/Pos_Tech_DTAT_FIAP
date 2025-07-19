import pandas as pd
import numpy as np
import streamlit as st
import pickle
import requests
import spacy
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide",
                     page_title="Análise de Similaridade de Currículos e Vagas"
                   )

url = "https://meuarquivo.blob.core.windows.net/modelo/modelo_ml.pkl"
response = requests.get(url)

# Carrega modelo spaCy para similaridade
nlp = spacy.load("pt_core_news_md")

def get_similarity(txt1, txt2):
    if pd.isna(txt1) or pd.isna(txt2): return 0.0
    doc1, doc2 = nlp(str(txt1)), nlp(str(txt2))
    if doc1.has_vector and doc2.has_vector:
        return doc1.similarity(doc2)
    return 0.0

# === Carregamento dos dados ===

vagas_url = "https://57datathon.blob.core.windows.net/data/processed/vagas/vagas.csv"

df_vagas = pd.read_csv(
    vagas_url,
    encoding='utf-8'
)

# Cria coluna texto_vaga_completo juntando todas as informações relevantes da vaga
df_vagas['texto_vaga_completo'] = (
    df_vagas['informacoes_basicas_titulo_vaga'].fillna('') + ' ' +
    df_vagas['perfil_vaga_cidade'].fillna('') + ' ' +
    df_vagas['perfil_vaga_estado'].fillna('') + ' ' +
    df_vagas['perfil_vaga_pais'].fillna('') + ' ' +
    df_vagas['perfil_vaga_nivel profissional'].fillna('') + ' ' +
    df_vagas['perfil_vaga_nivel_academico'].fillna('') + ' ' +
    df_vagas['perfil_vaga_nivel_ingles'].fillna('') + ' ' +
    df_vagas['perfil_vaga_nivel_espanhol'].fillna('') + ' ' +
    df_vagas['perfil_vaga_areas_atuacao'].fillna('') + ' ' +
    df_vagas['perfil_vaga_principais_atividades'].fillna('') + ' ' +
    df_vagas['perfil_vaga_competencia_tecnicas_e_comportamentais'].fillna('')# + ' ' +
    #df_vagas['perfil_vaga_competencia_tecnicas_e_comportamentais_pt'].fillna('')
).str.replace(r'\n+', ' ', regex=True).str.strip()

vagas = df_vagas['informacoes_basicas_titulo_vaga'].unique().tolist()

# === Interface ===
st.title("Análise de Similaridade de Currículos e Vagas")
st.write("Este aplicativo permite analisar a similaridade entre currículos e vagas de emprego usando modelos de Machine Learning.")

vaga_selecionada = st.selectbox("Selecione a vaga de interesse:", vagas, key="vaga_selecionada")
resultado = df_vagas[df_vagas['informacoes_basicas_titulo_vaga'] == vaga_selecionada]

if not resultado.empty:
    st.subheader("Detalhes da Vaga Selecionada")
    st.text(resultado['perfil_vaga_principais_atividades'].values[0])
    st.text(resultado['perfil_vaga_competencia_tecnicas_e_comportamentais'].values[0])
else:
    st.warning("Vaga sem descrição.")

#if st.button("Verificar sua compatibilidade com a vaga"):
st.write("Para verificar a compatibilidade, preencha os campos abaixo com as informações do seu currículo.")
# === Campos do Currículo ===
nome = st.text_input("Nome Completo:", key="nome")
telefone = st.text_input("Telefone:", key="telefone")
email = st.text_input("E-mail:", key="email")
cv_pt = st.text_input("Insira o texto do currículo:", key="cv_text")
objetivo = st.text_input("Objetivo Profissional:", key="objetivo")
experiencia = st.text_area("Experiência Profissional:", key="experiencia")
cursos = st.text_area("Cursos:", key="cursos")
qualificacoes = st.text_area("Qualificações:", key="qualificacoes")
nivel_profissional = st.selectbox("Nível Profissional:", ["Nenhum", "Júnior", "Pleno", "Sênior", "Especialista", "Gerente", "Diretor"], key="nivel_profissional")
conhecimentos_tecnicos = st.text_area("Conhecimentos Técnicos:", key="conhecimentos_tecnicos")
escolaridade = st.selectbox("Escolaridade:", ["Nenhum", "Ensino Médio", "Graduação", "Pós-Graduação", "Mestrado", "Doutorado"], key="escolaridade")
ingles = st.selectbox("Nível de Inglês:", ["Nenhum", "Básico", "Intermediário", "Avançado", "Fluente"], key="ingles")
espanhol = st.selectbox("Nível de Espanhol:", ["Nenhum", "Básico", "Intermediário", "Avançado", "Fluente"], key="espanhol")

# === Descrição da vaga para similaridade ===
descricao_vaga_selecionada = resultado['texto_vaga_completo'].values[0]

# === Quando o botão for pressionado ===
if st.button("Calcular Similaridade"):
    if not cv_pt:
        st.error("Por favor, insira o texto do currículo.")
    else:
        # Carrega o modelo
        url = "https://meuarquivo.blob.core.windows.net/modelo/modelo_ml.pkl"
        response = requests.get(url)
        pipeline = pickle.loads(response.content)


        # Cria DataFrame com dados do usuário
        data = {
            'cv_pt': [cv_pt],
            'infos_basicas_objetivo_profissional': [objetivo],
            'informacoes_profissionais_experiencias': [experiencia],
            'formacao_e_idiomas_cursos': [cursos],
            'informacoes_profissionais_qualificacoes': [qualificacoes],
            'informacoes_profissionais_nivel_profissional': [nivel_profissional],
            'informacoes_profissionais_conhecimentos_tecnicos': [conhecimentos_tecnicos],
            'escolaridade': [escolaridade],
            'ingles': [ingles],
            'espanhol': [espanhol]
        }
        df_input = pd.DataFrame(data)

        # === Similaridade entre currículo e vaga ===
        similaridade = get_similarity(descricao_vaga_selecionada, cv_pt)
        df_input['score_similaridade'] = [similaridade]

        # Dados usados no modelo
        df_ml = df_input[['cv_pt', 'escolaridade', 'ingles', 'espanhol', 'score_similaridade']]

        # === Mapas de categorias para números (se necessário) ===
        escolaridade_map = {
            'ensino fundamental incompleto': 0,
            'ensino fundamental cursando': 1,
            'ensino fundamental completo': 2,
            'ensino médio incompleto': 3,
            'ensino médio cursando': 4,
            'ensino médio completo': 5,
            'ensino técnico incompleto': 6,
            'ensino técnico cursando': 7,
            'ensino técnico completo': 8,
            'ensino superior incompleto': 9,
            'ensino superior cursando': 10,
            'ensino superior completo': 11,
            'pós graduação incompleto': 12,
            'pós graduação cursando': 13,
            'pós graduação completo': 14,
            'mestrado incompleto': 15,
            'mestrado cursando': 16,
            'mestrado completo': 17,
            'doutorado incompleto': 18,
            'doutorado cursando': 19,
            'doutorado completo': 20
        }

        idioma_map = {
            'nenhum': 0,
            'básico': 1,
            'intermediário': 2,
            'avançado': 3,
            'fluente': 4
        }

        df_ml['escolaridade'] = df_ml['escolaridade'].map(escolaridade_map)
        df_ml['ingles'] = df_ml['ingles'].map(idioma_map)
        df_ml['espanhol'] = df_ml['espanhol'].map(idioma_map)

        text_col = 'cv_pt'
        num_cols = ['escolaridade', 'ingles', 'espanhol', 'score_similaridade']

        df_ml['score_similaridade'] *= 3

        classe_1 = [
            'Aprovado',
            'Contratado como Hunting',
            'Contratado pela Decision',
            'Encaminhar Proposta',
            'Proposta Aceita'
        ]

        df_ml[text_col] = df_ml[text_col].fillna('')
        df_ml[num_cols] = df_ml[num_cols].fillna(0)

        # === Previsão ===
        prediction = pipeline.predict(df_ml)

        st.write(f"Similaridade com a vaga: **{similaridade * 100:.2f}%**")
        
        if prediction[0] == 1 or similaridade >= 0.7:
            st.success("✅ O currículo é compatível com a vaga selecionada.")
            st.link_button('Entrar em contato com o recrutador', url='https://www.linkedin.com/company/decisionbr-consultants/?originalSubdomain=br')
        else:
            st.error("❌ O currículo não é compatível com a vaga selecionada.")
