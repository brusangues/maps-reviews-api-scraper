import time
import re
import pandas as pd
from tqdm import tqdm

from llms.models import query_model

EVALUATION_PROMPT_TEMPLATE = """

[Sistema]
Por favor, atue como um juiz imparcial e avalie a qualidade das respostas fornecidas por dois assistentes de IA para a pergunta do usuário exibida abaixo. Sua avaliação deve considerar a correção e a utilidade das respostas. Você receberá a resposta do Assistente A e a resposta do Assistente B. Seu trabalho é avaliar qual resposta é melhor.  

Primeiro, resolva a pergunta do usuário passo a passo de forma independente. Em seguida, compare as respostas dos dois assistentes com a sua. Identifique e corrija quaisquer erros.  

Evite qualquer viés de posição e assegure-se de que a ordem de apresentação das respostas não influencie sua decisão. Não permita que o tamanho das respostas influencie sua avaliação. Não favoreça certos nomes de assistentes. Seja o mais objetivo possível.  

Após fornecer uma breve explicação, apresente seu veredito final seguindo estritamente este formato: `"[[A]]"` se a resposta do Assistente A for melhor, `"[[B]]"` se a resposta do Assistente B for melhor, e `"[[C]]"` em caso de empate.
Lembre-se de manter a explicação concisa e direto ao ponto.

[Pergunta do Usuário]  
{question}  
[Início da Resposta do Assistente A]  
{answer_a}  
[Fim da Resposta do Assistente A]  
[Início da Resposta do Assistente B]  
{answer_b}  
[Fim da Resposta do Assistente B]

"""

QUESTION = """
Você é um assistente de sumarização de hotéis em português.
Siga as INSTRUÇÕES do usuário para escrever um resumo detalhado do que se pede.
Faça um resumo do hotel com base nos seguintes tópicos:
1. Infraestrutura e Acomodações – Conforto, limpeza, tecnologia, lazer, estacionamento.
2. Atendimento e Serviço – Cordialidade, eficiência, limpeza, concierge, check-in ágil.
3. Localização e Acessibilidade – Proximidade, transporte, segurança, acessibilidade.
4. Alimentação e Bebidas – Café da manhã, restaurante, serviço de quarto, qualidade.
5. Experiência e Entretenimento – Lazer, eventos, recreação, passeios, parcerias.
6. Custo-benefício e Políticas – Preço justo, flexibilidade, transparência, fidelidade.
Para cada tópico, escreva um parágrafo com os principais aspectos positivos e negativos do hotel com base em suas avaliações.
"""


def pairwise_eval_series(s1, s2, question=QUESTION, llm=None):
    print("pairwise_eval_series...")

    evals = []
    for i in tqdm(range(len(s1)), total=len(s1), desc="pairwise_evals"):
        assert s1.index[i] == s2.index[i]
        data = pairwise_eval(
            s1=s1.iloc[i],
            s2=s2.iloc[i],
            s1_name=s1.name,
            s2_name=s2.name,
            nome=s1.index[i],
            question=question,
            llm=llm,
        )
        evals.append(data)
    return evals


def pairwise_eval(s1, s2, s1_name="", s2_name="", nome="", question=QUESTION, llm=None):
    # print("pairwise_eval...")

    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        question=question,
        answer_a=s1,
        answer_b=s2,
    )
    response, info = query_model(llm, prompt)

    try:
        if re.search(r"\[{2}[Aa]\]{2}", response):
            result = 0
        elif re.search(r"\[{2}[Bb]\]{2}", response):
            result = 1
        elif re.search(r"\[{2}[Cc]\]{2}", response):
            result = 0.5
    except Exception as e:
        print(e)
        result = None

    data = {
        "pairwise_eval": result,
        "s1_name": s1_name,
        "s2_name": s2_name,
        "nome": nome,
        "prompt": prompt,
        "response": response,
        "info": info,
    }
    return data
