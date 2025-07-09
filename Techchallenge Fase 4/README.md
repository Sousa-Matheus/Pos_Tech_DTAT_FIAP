# Tech Challenge 4 - 7DTAT FIAP (Grupo 34)

## 📌 Apresentação do Projeto

Este repositório contém o desenvolvimento do Tech Challenge da Fase 4 da pós-graduação em Data Analytics (FIAP), realizado pelo **Grupo 34**. O projeto tem como foco a ingestão, tratamento, modelagem preditiva e visualização de dados relacionados ao preço do petróleo Brent.

## 🧠 Objetivo

Criar um pipeline completo de ciência de dados, desde a ingestão de dados brutos até a apresentação de insights por meio de dashboards interativos, com foco na previsão dos valores futuros do petróleo Brent.

---

## ☁️ Arquitetura e Tecnologias Utilizadas

- **Azure Data Lake**: utilizado para armazenar os dados tratados.
- **ipeadatapy**: biblioteca utilizada para acessar a base de dados do [IPEA Data](http://www.ipeadata.gov.br/Default.aspx).
- **Python**: linguagem principal do projeto.
- **Pandas, NumPy**: para manipulação e análise dos dados.
- **Prophet**: biblioteca de modelagem preditiva para séries temporais.
- **Dash (Plotly)**: para visualização interativa dos resultados.
- **GitHub Actions**: para automação do deploy do modelo em ambiente Azure.

---

## 🔄 Pipeline do Projeto

1. **Ingestão de Dados**
   - Uso da biblioteca `ipeadatapy` para coletar dados econômicos do portal IPEA.
   - Scripts organizados no notebook `tratamento_ingestao_dados_ipea.ipynb`.

2. **Armazenamento**
   - Após o tratamento, os dados foram salvos no **Azure Data Lake**, garantindo persistência e disponibilidade.

3. **Modelagem**
   - Modelos de previsão foram criados com a biblioteca `Prophet`, descritos no notebook `modelo_prophet.ipynb`.

4. **Deploy**
   - O modelo treinado foi preparado para produção em `deploy_modelo.py`.
   - Deploy automatizado na **Azure Cloud** por meio de **GitHub Actions**, promovendo integração contínua e facilidade de atualização.

5. **Dashboard**
   - Interface web interativa construída com **Dash**, disponível no script `dash-petroleo-brent.py`.

---

## 📁 Estrutura do Repositório

| Arquivo | Descrição |
|--------|-----------|
| `tratamento_ingestao_dados_ipea.ipynb` | Coleta e tratamento dos dados via ipeadatapy e salvamento no Azure Data Lake. |
| `modelo_prophet.ipynb` | Modelagem preditiva usando Prophet para prever os preços do petróleo. |
| `deploy_modelo.py` | Script para deploy do modelo treinado na Azure. |
| `dash-petroleo-brent.py` | Construção do dashboard para visualização interativa das previsões. |
| `techchallenge4.ipynb` | Notebook integrador, conectando todas as etapas do projeto. |
| `requirements.txt` | Lista de dependências do projeto. |
| `LEIAME.md` | Documento explicativo e descritivo do projeto. |

---

## 🚀 Como Executar o Projeto

1. Clone o repositório:
   ```bash
   git clone https://github.com/Sousa-Matheus/Pos_Tech_DTAT_FIAP.git
   cd Pos_Tech_DTAT_FIAP/Techchallenge Fase 4

