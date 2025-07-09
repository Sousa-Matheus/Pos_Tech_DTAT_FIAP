# === 1. Importações ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.compose import ColumnTransformer
import spacy

# Carrega modelo do spaCy para similaridade semântica
nlp = spacy.load("pt_core_news_md")

# === 2. Carregamento dos dados ===
df_candidatos = pd.read_csv("C:/Users/Mathw/Documents/GitHub/Pos_Tech_DTAT_FIAP/Datathon Fase 5/data/processed/candidatos.csv", encoding='utf-8')
df_vagas = pd.read_csv("C:/Users/Mathw/Documents/GitHub/Pos_Tech_DTAT_FIAP/Datathon Fase 5/data/processed/vagas.csv", encoding='utf-8')
df_prospect = pd.read_csv("C:/Users/Mathw/Documents/GitHub/Pos_Tech_DTAT_FIAP/Datathon Fase 5/data/processed/prospect.csv", encoding='utf-8')

# === 3. Monta textos compostos ===
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
    df_vagas['perfil_vaga_competencia_tecnicas_e_comportamentais'].fillna('')
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

# === 4. Merge dos dados ===
df_ml = df_prospect.copy()
df_ml = df_ml[df_ml['prospects_situacao_candidado'].notna()]
df_ml = df_ml.merge(df_candidatos, left_on='prospects_nome', right_on='infos_basicas_nome', how='left')
df_ml = df_ml.merge(df_vagas, left_on='titulo', right_on='informacoes_basicas_titulo_vaga', how='left')

# === 5. Mapeamentos ===
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

def get_similarity(txt1, txt2):
    if pd.isna(txt1) or pd.isna(txt2): return 0.0
    doc1, doc2 = nlp(str(txt1)), nlp(str(txt2))
    if doc1.has_vector and doc2.has_vector:
        return doc1.similarity(doc2)
    return 0.0

# === 6. Feature engenharia ===
df_ml['score_similaridade'] = df_ml.apply(
    lambda row: get_similarity(row['cv_pt_info_profissionais'], row['texto_vaga_completo']), axis=1
)

df_ml['escolaridade'] = df_ml['formacao_e_idiomas_nivel_academico'].str.lower().map(escolaridade_map)
df_ml['ingles'] = df_ml['formacao_e_idiomas_nivel_ingles'].str.lower().map(idioma_map)
df_ml['espanhol'] = df_ml['formacao_e_idiomas_nivel_espanhol'].str.lower().map(idioma_map)

# === 7. Define features e target ===
text_col = 'cv_pt'
num_cols = ['escolaridade', 'ingles', 'espanhol', 'score_similaridade']

df_ml[text_col] = df_ml[text_col].fillna('')
X = df_ml[[text_col] + num_cols].fillna(0)
y = df_ml['prospects_situacao_candidado'].fillna("Indefinido")
y = y.apply(lambda x: 1 if x.lower() == 'aprovado' else 0)

# === 8. Pré-processamento e pipeline ===
preprocessor = ColumnTransformer([
    ('tfidf', TfidfVectorizer(max_features=100), text_col),
    ('num', StandardScaler(), num_cols)
])

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

# === 9. Grid Search ===
param_grid = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [None, 10],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2]
}

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2
)

# === 10. Treinamento e avaliação ===
grid_search.fit(X_train, y_train)
print("Melhores Hiperparâmetros: ", grid_search.best_params_)

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
