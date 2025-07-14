import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import spacy
import warnings
from tqdm import tqdm
tqdm.pandas()
warnings.filterwarnings("ignore")

df_ml = pd.read_csv("C:/Users/Mathw/Documents/GitHub/Pos_Tech_DTAT_FIAP/Datathon Fase 5/data/processed/modelo_ml.data.csv", encoding='utf-8')

# === Features e dados ===
text_col = 'cv_pt'
num_cols = ['escolaridade', 'ingles', 'espanhol', 'score_similaridade']

X = df_ml[[text_col]].fillna('')
X[num_cols] = df_ml[num_cols].fillna(0)
y = df_ml['prospects_situacao_candidado'].str.lower().isin([
    'aprovado', 'contratado como hunting', 'contratado pela decision',
    'encaminhar proposta', 'proposta aceita'
]).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === Pré-processamento ===
preprocessor = ColumnTransformer([
    ('tfidf', TfidfVectorizer(max_features=2000), text_col),
    ('num', StandardScaler(), num_cols)
])

# === Modelos para comparar ===
modelos = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(scale_pos_weight=5, use_label_encoder=False, eval_metric='logloss', random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
}

# === Loop para treinar e avaliar ===
resultados = []

for nome, modelo in modelos.items():
    print(f'\n=== Treinando {nome} ===')
    
    pipeline = ImbPipeline([
        ('preprocess', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', modelo)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    resultados.append({
        'Modelo': nome,
        'Accuracy': round(acc, 3),
        'Precision (Classe 1)': round(report['1']['precision'], 3),
        'Recall (Classe 1)': round(report['1']['recall'], 3),
        'F1-score (Classe 1)': round(report['1']['f1-score'], 3)
    })

# === Exibe resultados ===
df_resultados = pd.DataFrame(resultados)
print('\n\n=== Comparação Final ===')
print(df_resultados.to_string(index=False))