# 🧠 Seleção Inteligente de Candidatos — Datathon Fase 5

Este projeto tem como objetivo auxiliar o processo de recrutamento, aplicando técnicas de NLP (Processamento de Linguagem Natural) e Machine Learning para avaliar a compatibilidade entre candidatos e vagas de forma automatizada.

---

## 📁 Estrutura do Projeto

```plaintext
Datathon_Fase_5/
├── eda/
│   └── eda_elt_candidatos_decision.ipynb
├── modelo/
│   ├── .gitattributes
│   ├── modelo_ml.joblib
│   ├── nlp_comparacao.py
│   ├── randomforestclassifier_modelo.py
│   └── testar_modelos.py
├── utils/
│   ├── __pycache__/
│   ├── __init__.py
│   └── traduzir_ing_pt.py
├── .gitattributes
├── README.md
├── app.py
└── requirements.txt
```

---

## 🧪 Descrição do Projeto

1. **Coleta e Pré-processamento:**
   - Os dados foram extraídos e normalizados a partir de arquivos `.json`.
   - Após um breve EDA (Análise Exploratória dos Dados), os dados foram tratados e salvos em formato `.csv` em containers do Azure Storage Account.

2. **Processamento NLP:**
   - Campos como descrição das vagas e informações dos candidatos foram analisados com a biblioteca **spaCy**, atribuindo uma **nota de similaridade (0–100)** para cada par candidato-vaga.

3. **Criação da Base Prospect:**
   - As bases de **vagas** e **candidatos** foram integradas por meio de um `join` com a base **prospect**.
   - A base final foi salva em outro container no Azure.

4. **Modelo de Classificação:**
   - Foi executado um script de testes para avaliar diferentes modelos de classificação.
   - O melhor modelo foi treinado novamente com a base final, serializado com **Joblib** e armazenado no Azure.

5. **Aplicação Streamlit:**
   - A aplicação permite:
     - Selecionar uma vaga.
     - Cadastrar um novo candidato.
     - Calcular a similaridade usando o modelo NLP e o modelo de classificação.
   - Regras de aprovação:
     - Se a similaridade NLP ≥ 70%, o candidato é aprovado.
     - Se < 70%, o modelo pode aprovar com base em outras características (ex: escolaridade).
   - O app foi deployado no **Streamlit Cloud** e no **Azure Web App** como redundância.

---

## 🚀 Como Executar o Projeto

1. Clone o repositório:
   ```bash
   git clone https://github.com/Sousa-Matheus/Pos_Tech_DTAT_FIAP.git
   cd Pos_Tech_DTAT_FIAP/Datathon_Fase_5
   ```

2. Crie um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # ou venv\Scripts\activate no Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Execute o aplicativo:
   ```bash
   streamlit run app.py
   ```

---

## 🧠 Tecnologias Utilizadas

- Python
- Streamlit
- spaCy
- scikit-learn
- Joblib
- Azure Storage Account
- Jupyter Notebook

---

## 🔗 Deploys

- [Aplicação no Streamlit Cloud](#)
- [Backup no Azure Web App](#)

---

## 👥 Equipe

Projeto desenvolvido como parte do **Datathon Fase 5 — FIAP Pós-Tech em Data Analytics**.
