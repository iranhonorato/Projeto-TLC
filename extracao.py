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



# class DesafiosODS(Enum):
#     ODS_01 = "Erradicação da pobreza"
#     ODS_02 = "Fome zero e agricultura sustentável"
#     ODS_03 = "Saúde e bem-estar"
#     ODS_04 = "Educação de qualidade"
#     ODS_05 = "Igualdade de gênero"
#     ODS_06 = "Água potável e saneamento"
#     ODS_07 = "Energia limpa e acessível"
#     ODS_08 = "Trabalho decente e crescimento econômico"
#     ODS_09 = "Indústria, inovação e infraestrutura"
#     ODS_10 = "Redução das desigualdades"
#     ODS_11 = "Cidades e comunidades sustentáveis"
#     ODS_12 = "Consumo e produção sustentáveis"
#     ODS_13 = "Ação contra a mudança global do clima"
#     ODS_14 = "Vida na água"
#     ODS_15 = "Vida terrestre"
#     ODS_16 = "Paz, justiça e instituições eficazes"
#     ODS_17 = "Parcerias e meios de implementação"


# class ODSNumber(Enum):
#     ODS_01 = 1
#     ODS_02 = 2
#     ODS_03 = 3
#     ODS_04 = 4
#     ODS_05 = 5
#     ODS_06 = 6
#     ODS_07 = 7
#     ODS_08 = 8
#     ODS_09 = 9
#     ODS_10 = 10
#     ODS_11 = 11
#     ODS_12 = 12
#     ODS_13 = 13
#     ODS_14 = 14
#     ODS_15 = 15
#     ODS_16 = 16
#     ODS_17 = 17



# class EtapaCicloPP(Enum):
#     DEFINICAO = "Definição e Dimensão"
#     MOBILIZACAO = "Mobilização"
#     MAPEAMENTO = "Mapeamento dos Determinantes"
#     SOLUCAO = "Solução"
#     JUSTIFICATIVA = "Justificativa"
#     APRIMORAMENTO = "Aprimoramento"
#     CERTIFICACAO = "Certificação"


class Classificacao(Enum):
    ACADEMICA = "Acadêmica"
    TECNICA = "Técnica"


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
    ods: str = Field(..., description="Número das ODS relacionadas ao trabalho (texto)")
    etapa: str = Field(..., description="Etapa do ciclo de políticas públicas")
    titulo:str = Field(..., description="Titulo do trabalho")


def extracao_langchain(texto: str) -> Trabalho:
    system_text = """
    Você é um pesquisador especializado em políticas públicas e nos Objetivos de Desenvolvimento Sustentável (ODS) da ONU. 
    Seu papel é analisar textos, artigos e relatórios a partir de uma perspectiva acadêmica e técnica, utilizando uma abordagem baseada em evidências. 
    Sempre que receber um texto ou estudo, realize a seguinte tarefa: Responda **apenas** com um objeto JSON válido que tenha as seguintes chaves: 
    - classificacao: "Academica" (maior rigor acadêmico nos métodos e na escrita) ou "Tecnica" (ênfase à execução, procedimentos e resultados práticos, mais do que à fundamentação teórica) 
    - metodologia: "Qualitativa", "Quantitativa" ou "Mista" 
    - area_avaliada: "Educação", "Saúde", "Meio Ambiente", "Gênero", "Raça", "Pobreza" ou "Desenvolvimento Social" 
    - ods: o número das ODS's (Objetivo de Desenvolvimento Sustentável da ONU) relacionadas ao trabalho no formato textual (Podem ter mais de uma relacionadas)
    - etapa do ciclo de políticas públicas (Podem ter mais de uma relacionadas): 
        1. "Definição e Dimensão": É o ponto de partida de toda política pública. Consiste em especificar claramente o problema social que se deseja resolver, medir sua magnitude e suas consequências. Inclui três subetapas:
        Especificação e mensuração: definir o problema e os resultados de interesse.
        Magnitude e evolução: avaliar o tamanho do problema e sua evolução no tempo e entre grupos.
        Consequências: estimar o que acontece se nada for feito e quais impactos uma intervenção pode gerar. 
        2. "Mobilização": Toda política depende de apoio e ação coletiva.
        Nessa etapa, busca-se engajar os atores-chave e a sociedade, entendendo seus interesses e percepções sobre o problema. Inclui:
        Grau de sensibilização e mobilização;
        Percepção dos atores-chave;
        Eficácia da mobilização.
        O foco é garantir apoio social e político para a política pública
        3."Mapeamento dos Determinantes": Aqui identifica-se as causas do problema, os fatores que influenciam os resultados de interesse.
        O objetivo é mapear e priorizar os determinantes sobre os quais a política pode agir com mais impacto. Subetapas:
        Mapa de determinantes;
        Priorização.
        É um diagnóstico fundamentado para orientar a escolha das ações mais eficazes
        4."Solução": Corresponde ao desenho da política pública.
        Com base nos determinantes e nas evidências, busca-se definir as estratégias de intervenção e construir um modelo de mudança, ou seja, como a ação leva ao resultado. Inclui:
        Estratégia de solução e modelo de mudança;
        Validade do modelo de mudança;
        Compromissos e metas.
        Aqui são definidos os objetivos concretos e as metas mensuráveis
        5. "Justificativa": Avalia se a política é viável e vale o investimento.
        Baseia-se na análise da relação entre custos e benefícios esperados, antes da implementação (análise ex-ante). Inclui:
        Impacto (eficácia esperada);
        Valoração do custo e dos benefícios;
        Relações custo-benefício e custo-efetividade.
        A solução é considerada justificada se seus benefícios superam seus custos 
        6. "Aprimoramento": Trata-se do processo contínuo de melhoria da política durante sua implementação.
        Usa-se a evidência gerada pelo monitoramento para ajustar o modelo e o modo de operação. Inclui:
        Monitoramento;
        Eficiência (uso dos recursos);
        Eficácia (impacto real obtido);
        Validação do modelo de mudança e do modo de operação.
        O objetivo é aprender com a prática e melhorar continuamente
        7. "Certificacao": Etapa final de avaliação ex-post.
        Serve para prestar contas e consolidar o aprendizado para políticas futuras. Inclui:
        Eficiência;
        Eficácia;
        Relações custo-benefício e custo-efetividade;
        Adequação (a solução atendeu bem à população-alvo?).
        A certificação assegura que a política alcançou seus objetivos dentro do orçamento e gera evidências para outras iniciativas 
    - titulo: Extraia o título do trabalho (Apenas copie e cole o título. Não precisa traduzir)

    Responda somente com um JSON **válido**, utilizando **aspas duplas** em todas as chaves e valores, e nada mais. O formato deve ser exatamente este:

    {{
    "classificacao": "...",
    "metodologia": "...",
    "area_avaliada": "...",
    "ods": "...",
    "etapa": "...",
    "titulo": "..."
    }}

    
    """ 

    user_template = '''
    Trabalho:
    {texto}

    Baseie-se nos exemplos abaixo para fazer a análise do trabalho acima:
    Exemplo 1: 
    O conteúdo hospedado na seguinte URL: https://periodicos.fgv.br/rbe/article/view/84448
    
    'classificacao': 'Acadêmica',
    'metodologia': 'Quantitativa',
    'area_avaliada': 'Meio Ambiente'
    'ods': '3; 11',
    'etapa': 'Certificação', 
    'titulo': 'When streets have no men: Urban traffic restrictions and air pollution'

    Exemplo 2: 
    O conteúdo hospedado na seguinte URL: https://www.sciencedirect.com/science/article/abs/pii/S0165176523005281?via%3Dihub
    
    'classificacao': 'Acadêmica',
    'metodologia': 'Quantitativa',
    'area_avaliada': 'Desenvolvimento Social'
    'ods': '2; 8; 11',
    'etapa': 'Mapeamento de Determinantes', 
    'titulo': 'Technical change in agriculture and homicides: The case of genetically-modified soy seeds in Brazil'

    Exemplo 3: 
    O conteúdo hospedado na seguinte URL: https://repositorio.insper.edu.br/entities/publication/1e8eab33-99e1-4a71-89c4-290b5044a4f7
    
    'classificacao': 'Acadêmica',
    'metodologia': 'Mista',
    'area_avaliada': 'Educação'
    'ods': '3; 4; 10; 11',
    'etapa': 'Mapeamento dos Determinantes; Definição e Dimensão; Certificação.', 
    'titulo': 'Essays on Education and Health Economics'

    Exemplo 4:
    O conteúdo hospedado na seguinte URL: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0289604

    'classificacao': 'Acadêmica',
    'metodologia': 'Quantitativa',
    'area_avaliada': 'Saúde'
    'ods': '3',
    'etapa': 'Mapeamento dos Determinantes', 
    'titulo': 'The relationship between staying at home during the pandemic and the number of conceptions: A national panel data analysis'


    Exemplo 5: 
    O conteúdo hospedado na seguinte URL: https://ieps.org.br/panorama-ieps-01/
    'classificacao': 'Técnica', 
    'metodologia': 'Quantitativa', 
    'area_avaliada': 'Saúde', 
    'ods': '3; 11', 
    'etapa': 'Definição e Dimensão', 
    'titulo': 'Panorama da Cobertura Vacinal no Brasil, 2020'

    Exemplo 6:
    O conteúdo hospedado na seguinte URL: https://onlinelibrary.wiley.com/doi/10.1002/hec.4241
    'classificacao': 'Acadêmica', 
    'metodologia': 'Quantitativa', 
    'area_avaliada': 'Saúde', 
    'ods': '3; 8; 10', 
    'etapa': 'Definição e Dimensão', 
    'titulo': 'Financing needs, spending projection, and the future of health in Brazil'
    '''


    # Estruturação do prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("user", user_template)
    ])

    # Intância do modelo de linguagem utilizada
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3) 

    # Orienta o langchain a devolver uma reposta diretamente convertida em um objeto do tipo Trabalho
    structured_llm = llm.with_structured_output(Trabalho)

    # Encadeamento do texto com o prompt e a resposta estruturada que desejamos 
    chain = ({"texto": itemgetter("texto")} | prompt | structured_llm)

    try:
        # Invoca tudo: pega o seu texto, monta o prompt, chama o GPT e devolve um objeto (mas o encadeamento texto, prompt e structured_llm é feito pelo chain)
        # [System]: Você é um pesquisador especializado...
        # [User]: TEXTO: texto(passado no argumento)
        resposta = chain.invoke({"texto": texto})
        resultado = {
            "classificacao": resposta.classificacao.value,
            "metodologia": resposta.metodologia.value,
            "area_avaliada": resposta.area_avaliada.value,
            "ods": resposta.ods,
            "etapa": resposta.etapa,
            "titulo": resposta.titulo
        }
        
        # Se tudo der certo, `resposta` deve ser uma instância de Trabalho
        return resultado

        
    except Exception as e:
        raise RuntimeError(f"Erro ao invocar LangChain/LLM: {e}")




# if __name__ == "__main__":
#     exemplo = """
# https://onlinelibrary.wiley.com/doi/10.1002/sd.2754
#     """

#     resultado = extracao_langchain(exemplo)
#     print(resultado)

    