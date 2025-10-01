import os
from operator import itemgetter
from enum import Enum
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import Field, BaseModel





load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")







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




class EtapaCicloPP(Enum):
    AGENDA = "Reconhecimento de demandas sociais que precisam de atenção do Estado."
    ALTERNATIVAS = "Elaboração de propostas e possíveis soluções para o problema identificado."
    DECISAO = "Tomada de decisão. Escolha, entre as alternativas formuladas, da política a ser implementada."
    IMPLEMENTACAO = "Transformação da decisão em ação concreta."
    AVALIACAO = "Análise dos resultados, impactos e eficácia da política."





class Artigo(BaseModel):
    ods_relacionada: DesafiosODS = Field(..., description="O Objetivo de Desenvolvimento Sutentável (ODS) ao qual o artigo se trata.")
    resumo: str = Field(..., description="Resumo do artigo.")
    etapa: EtapaCicloPP = Field(..., description="Etapa do ciclo de políticas públicas do artigo em questão.")





