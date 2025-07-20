from langdetect import detect
from deep_translator import GoogleTranslator
import pandas as pd
import time

# Função para detectar idioma
def is_english(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except:
        return False

# Função para traduzir para português
def traduzir_para_pt(texto):
    try:
        return GoogleTranslator(source='auto', target='pt').translate(texto)
    except Exception as e:
        print(f"Erro ao traduzir: {e}")
        return texto

# Função geral para processar uma coluna inteira
def traduzir_coluna_se_ingles(df, coluna, nova_coluna):
    textos_traduzidos = []
    for texto in df[coluna]:
        if pd.isna(texto) or texto.strip() == "":
            textos_traduzidos.append("")
        elif is_english(texto):
            traduzido = traduzir_para_pt(texto)
            textos_traduzidos.append(traduzido)
            time.sleep(1)  # evitar limites da API gratuita
        else:
            textos_traduzidos.append(texto)
    df[nova_coluna] = textos_traduzidos
    return df
