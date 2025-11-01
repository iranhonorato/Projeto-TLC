# Agente CGPP 

## 1. Ferramentas utilizadas:

- Foi utilizado LangChain para o desenvolvimento do agente capaz de ler os trabalhos e interpretá-los, de modo a extrair as informações relevantes para a criação do Mapa de Evidências 

- Foi utilizado também a biblioteca pypdf para ler os PDFs dos trabalhos e, posteriormente, passá-los para o LangChain processar cada trabalho. 

**Todas as bibliotecas necessárias inicializar o projeto e utilizá-lo estão no requirements.txt**


## 2. Como rodar

Dois aquivos principais: **extracao.py** e **pdfs_validos.ipynb**

- Crie um aquivo venv e faça um pip install -r requirements.txt 

- Crie um arquivo .env e escreva a chave da API na seguinte forma: OPENAI_API_KEY = "{sua chave aqui}"

- No arquivo extracao.py encontra-se o código do agente 

- No arquivo pdfs_validos.ipynb importamos o arquivo excel da base de dados dos trabalhos do CGPP: 
**df_pdfs_validos = pd.read_excel("../Base 5.1.xlsx", sheet_name="PDFs_validos") (Atente-se ao fato de abrir especificamente o sheet de nome PDFs_validos)**

- Nesse mesmo arquivo pdfs_validos.ipynb encontra-se o código para extração dos textos dos PDFs dos trabalhos. Além disso, importamos o agente do arquivo extracao.py e o utilizamos para fazer a leitura dos PDFs 

