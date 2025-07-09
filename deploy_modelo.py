import streamlit as st
from datetime import datetime, timedelta
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go

# Layout do Streamlit
st.set_page_config(page_title="Previsão do Petróleo", layout="wide")
st.title("Previsão do valor do barril de petróleo")

hoje = datetime.today()
amanha = hoje + timedelta(days=1)

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.text_area('Data de hoje:', value=hoje.strftime("%d/%m/%Y"), height=68)

# Carregando dados do azure
url = "https://datalaketech4.blob.core.windows.net/dados-ipea/cotacao_petroleo_ipea.csv"

try:
    df_raw = pd.read_csv(url, encoding="ISO-8859-1", delimiter=";", decimal=",")
    df = df_raw.rename(columns={"data": "ds", "preco": "y"})[["ds", "y"]]
    df["ds"] = pd.to_datetime(df["ds"], dayfirst=True)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna()

    # Slider de previsão
    dias = st.slider("Escolha o número de dias para previsão", 7, 60, 30)

    # Treina modelo
    modelo = Prophet()
    modelo.fit(df)

    futuro = modelo.make_future_dataframe(periods=dias)
    previsao = modelo.predict(futuro)
    previsao['ds'] = pd.to_datetime(previsao['ds'], format='%d%m%y')
    # Filtra apenas os dias futuros
    ultima_data_real = df["ds"].max()
    previsao_futura = previsao[previsao["ds"] > ultima_data_real].head(dias)


    # Previsão para amanhã (hoje + 1 dia) - comparando apenas datas
    valor_amanha = previsao_futura[previsao_futura['ds'].dt.date == amanha.date()]['trend'].values


    # Verifica se existe previsão para amanhã
    if len(valor_amanha) > 0:
        valor_amanha = valor_amanha[0]
    else:
        valor_amanha = "Valor não disponível para amanhã"

    st.text_area('Previsão para Amanhã:', value=amanha.strftime("%d/%m/%Y"), height=68)
    st.text_area('Previsão do valor do barril de petróleo para amanhã:', value=f"R$ {valor_amanha:.2f}" if isinstance(valor_amanha, (float, int)) else valor_amanha, height=68)

    st.text_area('Gráfico de Previsão:', value="Abaixo está a previsão gerada para os próximos dias", height=68)

    # Gráfico filtrado apenas com a previsão futura
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=previsao_futura['ds'], y=previsao_futura['yhat'], name='Previsão'))
    fig.add_trace(go.Scatter(x=previsao_futura['ds'], y=previsao_futura['yhat_upper'], name='Limite superior', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=previsao_futura['ds'], y=previsao_futura['yhat_lower'], name='Limite inferior', line=dict(dash='dot')))
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Erro ao carregar os dados: {e}")
