import os
import json
from operator import itemgetter
from enum import Enum
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import Field, BaseModel, ValidationError





load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) 



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
    DEFINICAO = "Definição e Dimensão"
    MOBILIZACAO = "Mobilização"
    MAPEAMENTO = "Mapeamento dos Determinantes"
    SOLUCAO = "Solução"
    JUSTIFICATIVA = "Justificativa"
    APRIMORAMENTO = "Aprimoramento"
    CERTIFICACAO = "Certificacao"


class Classificacao(Enum):
    ACADEMICA = "Academica"
    TECNICA = "Tecnica"


class AreaAvaliada(Enum):
    EDUCACAO = "Educação"
    SAUDE = "Saúde"
    MEIO_AMBIENTE = "Meio Ambiente"
    GENERO = "Gênero"
    RACA = "Raça"
    POBREZA = "Pobreza"
    DESENVOLVIMENTO_SOCIAL = "Desenvolvimento Social"


class Metodologia(Enum):
    QUALITATIVA = "Qualitativa"
    QUANTITATIVA = "Quantitativa"
    MISTA = "Mista"


class Trabalho(BaseModel):
    classificacao: Classificacao = Field(..., description="Acadêmico ou técnico")
    metodologia: Metodologia = Field(..., description="Abordagem do trabalho")
    area_avaliada: AreaAvaliada = Field(..., description="Área avaliada no trabalho")
    ods_relacionada: DesafiosODS = Field(..., description="ODS relacionado ao trabalho (texto)")
    etapa: EtapaCicloPP = Field(..., description="Etapa do ciclo de políticas públicas")
    resumo: str = Field(..., description="Resumo do trabalho")


def extracao_langchain(texto: str, max_sentences: int = 8) -> Trabalho:

    system_text = f"""
    Você é um pesquisador especializado em políticas públicas e nos Objetivos de Desenvolvimento Sustentável (ODS) da ONU. 
    Seu papel é analisar textos, artigos e relatórios a partir de uma perspectiva acadêmica e técnica, utilizando uma abordagem baseada em evidências. 
    Sempre que receber um texto ou estudo, realize a seguinte tarefa: Responda **apenas** com um objeto JSON válido que tenha as seguintes chaves: 
    - classificacao: "Academica" ou "Tecnica" 
    - metodologia: "Qualitativa", "Quantitativa" ou "Mista" 
    - area_avaliada: "Educação", "Saúde", "Meio Ambiente", "Gênero", "Raça", "Pobreza" ou "Desenvolvimento Social" 
    - ods_relacionada: um dos 17 ODS, no formato textual (ex.: "Erradicação da pobreza") 
    - etapa: "Definição e Dimensão", "Mobilização", "Mapeamento dos Determinantes", "Solução", "Justificativa", "Aprimoramento" ou "Certificacao" 
    - resumo: Produza um resumo analítico, em até {max_sentences} sentenças, destacando as evidências, lacunas, relevância social e possíveis impactos da pesquisa/política. 
    Suas respostas devem ser claras, fundamentadas e conectadas com as melhores práticas de análise de políticas públicas baseadas em evidências.

    Responda somente com o JSON — nada mais.
    """


    user_template = "TEXTO:\n{texto}"


    # Estruturação do prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("user", user_template)
    ])

    # Intância do modelo de linguagem utilizada
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0) 

    # Orienta o langchain a devolver uma reposta diretamente convertida em um objeto do tipo Trabalho
    structured_llm = llm.with_structured_output(Trabalho)

    # Encadeamento do texto com o prompt e a resposta estruturada que desejamos 
    chain = ({"texto": itemgetter("texto")} | prompt | structured_llm)

    try:
        # Invoca tudo: pega o seu texto, monta o prompt, chama o GPT e devolve um objeto (mas o encadeamento texto, prompt e structured_llm é feito pelo chain)
        # [System]: Você é um pesquisador especializado...
        # [User]: TEXTO: texto(passado no argumento)
        resposta = chain.invoke({"texto": texto})

        # Se tudo der certo, `resposta` deve ser uma instância de Trabalho
        if isinstance(resposta, Trabalho):
            return resposta
        
        # em alguns setups o LangChain retorna dict; tente converter
        try:
            return Trabalho.model_validate(resposta)
        except ValidationError as ve:
            raise RuntimeError(f"Resposta do LLM recebida, mas inválida para 'Trabalho': {ve}")
        
        
    except Exception as e:
        # fallback e debug — mostra erro e relança
        raise RuntimeError(f"Erro ao invocar LangChain/LLM: {e}")




# -----------------------------
# Util: encoder para serializar Enums em JSON
# -----------------------------
def enum_encoder(o):
    if isinstance(o, Enum):
        return o.value
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


# -----------------------------
# Exemplo de uso
# -----------------------------
if __name__ == "__main__":
    exemplo = """
    Analyzing the quality factor for Brazil
    Abstract
    Investments in Brazil are increasingly allocated to the stock market, at the expense of more conservative investments. 
    Would finding higher-quality assets allow investors to increase their risk-return ratio? 
    We analyze quality with several metrics, including the quality-minus-junk (QMJ) factor for Brazil. 
    We find that quality companies are valued more by investors, with a higher price-to-book ratio. 
    A portfolio of shares of higher quality shows a significant positive return over the period analyzed, adjusted for several risk factors. 
    The sample members classified as quality companies remained within this classification over time.
    """

    resultado = extracao_langchain(exemplo)

    # imprimir JSON com Enums serializados
    data = resultado.model_dump()  # dict
    print(json.dumps(data, ensure_ascii=False, indent=2, default=enum_encoder))