# Web Scraping com Langchain e OpanAI API KEY


## Exemplo:


### Traz para o Python o módulo os, que serve para interagir com o sistema operacional.
**import os**
>> No seu projeto, o os será usado principalmente para pegar as chaves da OpenAI (ou de outros serviços) que você deixou no .env.


### Pegar valores específicos em tuplas, listas ou dicionários
**from operator import itemgetter**
>> O itemgetter é útil quando você precisa pegar valores específicos de uma tupla, lista ou dicionário.
>> Exemplo prático: você tem a saída do modelo como um dicionário com vários campos ({"resposta": "...", "tokens": 123}) e só quer o campo "resposta". O itemgetter("resposta") facilita.



### Usado para criar constantes nomeadas (valores fixos).
**from enum import Enum**
>> Exemplo prático: você pode definir um Enum para diferentes modos de scraping:
>> class ScrapingMode(Enum):
>>    HTML = "html"
>>    API = "api"
>>    PDF = "pdf"
>> Assim você evita ficar passando strings soltas pelo código e dá mais clareza.



### Para Carregar o arquivo .env e a OpenAI API KEY 
**from dotenv import load_dotenv** 
>> Exemplo prático: você coloca sua chave no .env:
>> OPENAI_API_KEY="minha_chave_aqui"
>> load_dotenv()
>> Pronto, agora o os.getenv("OPENAI_API_KEY") vai funcionar sem expor a chave no código.


### É a classe que conecta o LangChain ao chat da OpenAI (GPT-3.5, GPT-4, etc.).
**from langchain_openai import ChatOpenAI**
>> llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
>> e pode passar prompts para ele gerar respostas.



### Cria os templates de prompts compatíveis com os modelos de chat 
**from langchain_core.prompts import ChatPromptTemplate**
>> Usado para criar templates de prompts com variáveis dinâmicas.
>> Exemplo prático:
>> prompt = ChatPromptTemplate.from_template(
>>     "Resuma o seguinte texto em 3 linhas: {texto}"
>> )
>> Na hora de rodar:
>> resposta = llm.invoke(prompt.format(texto="um artigo grande"))


### Reserva o espaço para o histórico de mensagens no prompt 
**from langchain_core.prompts import MessagesPlaceholder**
>> Reserva um espaço no prompt para inserir histórico de mensagens.
>> Exemplo prático:
>> Você está fazendo um chat que precisa lembrar o que foi dito antes:
>> prompt = ChatPromptTemplate.from_messages([
>>     ("system", "Você é um assistente."),
>>     MessagesPlaceholder(variable_name="history"),
>>     ("human", "{nova_pergunta}")
>> ])
>>Assim o histórico entra automaticamente na conversa.


### Armazenam o histórico da nossa conversa em um formato estruturado 
**from langchain_core.runnables.history import RunnableWithMessageHistory**
>> Permite rodar um modelo junto com o histórico da conversa.
>> Exemplo prático:
>> Você transforma seu llm em um objeto que já sabe como lidar com histórico:
>> llm_with_history = RunnableWithMessageHistory(llm, ...)

**from langchain_core.chat_history import BaseChatMessageHistory**
>> É uma classe base para armazenar o histórico em memória, banco de dados ou qualquer outro lugar.
>> Exemplo prático: você pode implementar uma versão que guarda o histórico no Redis, em vez de só na memória.



## Serve para trabalhar com modelos de dados estruturados e validados no Python.
**from pydantic import Field, BaseModel**
>> No contexto de LangChain + OpenAI, ele é MUITO útil para garantir que a saída do modelo venha em um formato fixo (ex: JSON com campos obrigatórios).






load_dotenv(override=True)


class DesafiosODS(Enum):
    ODS_01 = "Erradicação da pobreza"
    ODS_02 = "Fome zero e agricultura sustentável"
    ODS_03 = "Saúde e bem-estar"
    ODS_04 = "Educação de qualidade"
    ODS_05 = "Igualdade de gênero"
    ODS_06 = "Água potável e saneamento"
    ODS_07 = "Energia limpa e acessível"
    ODS_08 = "Trabalho decente e crescimento econômico"
    ODS_09 = "Indústria, inovação e infraestrutura"
    ODS_10 = "Redução das desigualdades"
    ODS_11 = "Cidades e comunidades sustentáveis"
    ODS_12 = "Consumo e produção sustentáveis"
    ODS_13 = "Ação contra a mudança global do clima"
    ODS_14 = "Vida na água"
    ODS_15 = "Vida terrestre"
    ODS_16 = "Paz, justiça e instituições eficazes"
    ODS_17 = "Parcerias e meios de implementação"


class Artigo(BaseModel):
    valor: DesafiosODS = Field(..., description="O Desafio da ODS da ONU, no qual o artigo se trata.")
    resumo: str = Field(..., description="Resumo do artigo.")


def get_article_property(resumo: str) -> Artigo:

    model = ChatOpenAI(model="gpt-4o")
    structured_model = model.with_structured_output(Artigo)

    message = """
    prompt
    """


    prompt = ChatPromptTemplate.from_messages(
        [('system'), (message),
        ('user', "{resumo}")]
    )

    chain = (
        {"resumo": itemgetter("resumo")}
        | prompt
        | structured_model
    )

    resposta = chain.invoke({"resumo": resumo})

    return resposta