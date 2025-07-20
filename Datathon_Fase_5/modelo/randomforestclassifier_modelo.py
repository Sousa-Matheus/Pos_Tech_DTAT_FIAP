# === 1. Importações ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
from azure.storage.blob import BlobServiceClient
import warnings
from tqdm import tqdm
tqdm.pandas()
warnings.filterwarnings("ignore")

print('Bibliotecas importadas com sucesso!')

ml_data_url = "https://57datathon.blob.core.windows.net/data/processed/ml_data/modelo_ml.data.csv"

df_ml = pd.read_csv(ml_data_url, encoding='utf-8')

# === 7. Define features e target ===
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

X = df_ml[[text_col]].fillna('')
X[num_cols] = df_ml[num_cols].fillna(0)
y = df_ml['prospects_situacao_candidado'].fillna("Indefinido")

# Deixa a lista toda em minúsculas para padronizar
classe_1 = [c.lower() for c in classe_1]

# Converte a coluna para minúsculas e verifica se o valor está na lista
y = df_ml['prospects_situacao_candidado'].str.lower().isin(classe_1).astype(int)

print('Features e target definidos com sucesso!')

# === 8. Pré-processamento e pipeline ===
preprocessor = ColumnTransformer([
    ('tfidf', TfidfVectorizer(max_features=2000), text_col),
    ('num', StandardScaler(), num_cols)
])

pipeline = ImbPipeline([
    ('preprocess', preprocessor),
    #('undersample', RandomUnderSampler(random_state=42)),
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(random_state=42))
])

print('Pipeline de pré-processamento e modelo definida com sucesso!')

# === 9. Grid Search ===
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10],
    'model__min_samples_split': [2, 5],
    'model__class_weight': [None, 'balanced', 'balanced_subsample']
}

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring='precision',
    cv=3,
    n_jobs=-1,
    verbose=2
)

print('Grid Search definido com sucesso!')

# === 10. Treinamento e avaliação ===
grid_search.fit(X_train, y_train)
print("Melhores Hiperparâmetros: ", grid_search.best_params_)

print('Model treinado com sucesso!')

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === Matriz de confusão ===
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Não aprovado (0)', 'Aprovado (1)'])
disp.plot(cmap='Blues')
plt.title('Matriz de Confusão')
plt.show()

# === 11. Salvando o modelo ===
joblib.dump(best_model, 'modelo_ml.joblib', compress=3)
print("Modelo salvo com sucesso!")

# === 12. upload do modelo salvo para blob storage ===

connection_string = "DefaultEndpointsProtocol=https;AccountName=57datathon;AccountKey=V1EJu1EdJ/wJ+vPn5OkQ3ydwXQJaFfYfZp0thx1hJ/GPByONbi0U9WM+mdfELCvhAWJFgxDjdkrW+AStRI1IQQ==;EndpointSuffix=core.windows.net"
container_name = "modelo"
blob_name = "modelo_ml.joblib"

with open("modelo_ml.joblib", "rb") as data:
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_client.upload_blob(data, overwrite=True)