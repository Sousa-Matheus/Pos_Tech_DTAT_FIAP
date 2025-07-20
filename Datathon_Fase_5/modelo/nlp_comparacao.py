# === 1. Importações ===
import pandas as pd
import numpy as np
import spacy
import warnings
from io import BytesIO
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm
tqdm.pandas()
warnings.filterwarnings("ignore")

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.traduzir_ing_pt import traduzir_coluna_se_ingles


print('Bibliotecas importadas com sucesso!')

# Carrega modelo do spaCy para similaridade semântica
nlp = spacy.load("pt_core_news_md")

print('Modelo spaCy carregado com sucesso!')



# === 2. Carregamento dos dados ===

candidatos_url = "https://57datathon.blob.core.windows.net/data/processed/candidatos/candidatos.csv"
vagas_url = "https://57datathon.blob.core.windows.net/data/processed/vagas/vagas.csv"
prospect_url = "https://57datathon.blob.core.windows.net/data/processed/prospect/prospect.csv"

df_candidatos = pd.read_csv(candidatos_url, encoding='utf-8')
df_vagas = pd.read_csv(vagas_url, encoding='utf-8')
df_prospect = pd.read_csv(prospect_url, encoding='utf-8')

print('Dados carregados com sucesso!')

# === 3. Traduzir colunas de texto se necessário ===
# df_vagas['perfil_vaga_competencia_tecnicas_e_comportamentais_pt'] = traduzir_coluna_se_ingles(
#     df_vagas, 
#     'perfil_vaga_competencia_tecnicas_e_comportamentais', 
#     'perfil_vaga_competencia_tecnicas_e_comportamentais_pt'
# )

# === 4. Monta textos compostos ===
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
    df_vagas['perfil_vaga_competencia_tecnicas_e_comportamentais'].fillna('') + ' ' +
    df_vagas['perfil_vaga_competencia_tecnicas_e_comportamentais_pt'].fillna('')
)

df_vagas['texto_vaga_completo'] = df_vagas['texto_vaga_completo'].str.replace(r'\n+', ' ', regex=True).str.strip()

df_candidatos['texto_candidato_completo'] = (
    df_candidatos['cv_pt'].fillna('') + ' ' +
    df_candidatos['infos_basicas_objetivo_profissional'].fillna('') + ' ' +
    df_candidatos['infos_basicas_local'].fillna('') + ' ' +
    df_candidatos['infos_basicas_nome'].fillna('') + ' ' +
    df_candidatos['informacoes_profissionais_titulo_profissional'].fillna('') + ' ' +
    df_candidatos['informacoes_profissionais_conhecimentos_tecnicos'].fillna('') + ' ' +
    df_candidatos['informacoes_profissionais_nivel_profissional'].fillna('') + ' ' +
    df_candidatos['informacoes_profissionais_qualificacoes'].fillna('') + ' ' +
    df_candidatos['informacoes_profissionais_experiencias'].fillna('') + ' ' +
    df_candidatos['formacao_e_idiomas_nivel_academico'].fillna('') + ' ' +
    df_candidatos['formacao_e_idiomas_nivel_ingles'].fillna('') + ' ' +
    df_candidatos['formacao_e_idiomas_nivel_espanhol'].fillna('') + ' ' +
    df_candidatos['formacao_e_idiomas_cursos'].fillna('') + ' ' +
    df_candidatos['formacao_e_idiomas_outro_curso'].fillna('')
)

df_candidatos['texto_candidato_completo'] = df_candidatos['texto_candidato_completo'].str.replace(r'\n+', ' ', regex=True).str.strip()

print('Textos compostos criados com sucesso!')

# === 5. Merge dos dados ===
df_ml = df_prospect.copy()
df_ml = df_ml[df_ml['prospects_situacao_candidado'].notna()]
df_ml = df_ml.merge(df_candidatos, left_on='prospects_nome', right_on='infos_basicas_nome', how='left')
df_ml = df_ml.merge(df_vagas, left_on='titulo', right_on='informacoes_basicas_titulo_vaga', how='left')

print(f'Dados mesclados com sucesso!{df_ml.shape}')

# === 6. Mapeamentos ===
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

print('Mapeamentos criados com sucesso!')

# === 7. Função de similaridade ===

def get_similarity(txt1, txt2):
    if pd.isna(txt1) or pd.isna(txt2): return 0.0
    doc1, doc2 = nlp(str(txt1)), nlp(str(txt2))
    if doc1.has_vector and doc2.has_vector:
        return doc1.similarity(doc2)
    return 0.0

print('Função de similaridade definida com sucesso!')

print('Aplicando função de similaridade...')

# === 8. Feature engenharia ===
df_ml['score_similaridade'] = df_ml.progress_apply(
    lambda row: get_similarity(row['texto_candidato_completo'], row['texto_vaga_completo']), axis=1
)

print('Feature de similaridade calculada com sucesso!')

df_ml['escolaridade'] = df_ml['formacao_e_idiomas_nivel_academico'].str.lower().map(escolaridade_map)
df_ml['ingles'] = df_ml['formacao_e_idiomas_nivel_ingles'].str.lower().map(idioma_map)
df_ml['espanhol'] = df_ml['formacao_e_idiomas_nivel_espanhol'].str.lower().map(idioma_map)

print('Features de escolaridade e idiomas mapeadas com sucesso!')

# === 9. Salvando o DataFrame final ===

connection_string = "DefaultEndpointsProtocol=https;AccountName=57datathon;AccountKey=V1EJu1EdJ/wJ+vPn5OkQ3ydwXQJaFfYfZp0thx1hJ/GPByONbi0U9WM+mdfELCvhAWJFgxDjdkrW+AStRI1IQQ==;EndpointSuffix=core.windows.net"
container_name = "data"
blob_name = f"processed/ml_data/modelo_ml.data.csv"

csv_buffer = BytesIO()
df_ml.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

blob_client.upload_blob(csv_buffer, overwrite=True)

print('Dados preparados para o modelo salvos com sucesso!')