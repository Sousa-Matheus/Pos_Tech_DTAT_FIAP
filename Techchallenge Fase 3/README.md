# Projeto de Análise da PNAD COVID - Meses 9, 10 e 11 de 2020

## Descrição
Este projeto tem como objetivo a análise dos dados da PNAD COVID para os meses de setembro, outubro e novembro de 2020. O fluxo de processamento dos dados inclui a extração, transformação e análise dos dados utilizando diferentes tecnologias de Big Data e Machine Learning.

## Pipeline de Processamento
1. **Extração dos Dados:**
   - Os dados foram obtidos do site [Base dos Dados](https://basedosdados.org/).

2. **Processamento no Google BigQuery:**
   - Os dados foram processados utilizando SQL no Google BigQuery.
   - A query utilizada está armazenada no arquivo `Query.txt` dentro da pasta `Archives`.

3. **Transferência para Azure Data Lake Storage:**
   - Utilizamos o **Azure Data Factory** para copiar os dados do BigQuery para o **Azure Data Lake Storage**.
   - Os arquivos do Data Factory estão localizados na pasta `Archives`.

4. **Processamento com Python (Pandas):**
   - Os dados foram manipulados e analisados com a biblioteca **Pandas**.
   - O código para análise está no notebook `analise_pnad_coviTechChallenge 3v2.ipynb`.

5. **Machine Learning:**
   - Foi desenvolvido um modelo de Machine Learning para exploração dos dados.
   - O código do modelo está no notebook `ModeloClassificação.ipynb`.

## Estrutura do Repositório
```
Techchallenge Fase 3/
│── analise_pnad_covid.ipynb  # Notebook com a análise exploratória dos dados
│── modelo_pnad_covid.ipynb   # Notebook com o modelo de Machine Learning
│
├── Archives/                 # Pasta contendo arquivos auxiliares
│   ├── data_factory_files/   # Arquivos do Azure Data Factory
│   ├── query.txt             # Query usada no BigQuery
```

## Tecnologias Utilizadas
- **BigQuery** para consulta e processamento inicial dos dados
- **Azure Data Factory** para transferência dos dados
- **Azure Data Lake Storage** para armazenamento
- **Python (Pandas, Matplotlib, Seaborn)** para tratamento e análise dos dados
- **Jupyter Notebook** para execução da análise e modelo de Machine Learning

## Como Executar
1. Clone o repositório:
   ```bash
   git clone git clone https://github.com/Sousa-Matheus/Pos_Tech_DTAT_FIAP.git
   cd Pos_Tech_DTAT_FIAP/Techchallenge\ Fase\ 3
   ```
2. Instale as dependências necessárias:
   ```bash
   pip install pandas jupyter matplotlib seaborn sklearn
   ```
3. Abra os notebooks Jupyter:
   ```bash
   jupyter notebook
   ```
4. Execute os notebooks `analise_pnad_coviTechChallenge 3v2.ipynb` e `ModeloClassificação.ipynb` para visualizar a análise e o modelo.

## Observações

Projeto Desenvolvido para fins acadêmicos
.


