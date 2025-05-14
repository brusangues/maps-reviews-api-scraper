import time
import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import os

from llms.models import load_model, query_model


LOAD_PREP_SUMMARIES = False
LIMIT = None
JUDGE_MODEL = "gemini-2.5-flash"
SUMMARIES = [
    "summaries_v2_google-ip_gemini-1.5-flash-8b_1000",
    "summaries_v2_google-ip_gemini-2.0-flash-lite_1000",
    "summaries_v2_google-ip_gemini-2.0-flash_1000",
    "summaries_v2_gte-ip_gemini-1.5-flash-8b_1000",
    "summaries_v2_gte-ip_gemini-2.0-flash-lite_1000",
    "summaries_v2_gte-ip_gemini-2.0-flash_1000",
    "summaries_v2_google-ip_gemma3_101",
]

COMPARISONS = [
    # Comparando tipos de resumo
    {
        "comparison": "v3_2flash_v2_vs_v1",
        "summary_name_1": "summaries_v2_google-ip_gemini-2.0-flash_1000",
        "summary_type_1": "response_v2",
        "summary_name_2": "summaries_v2_google-ip_gemini-2.0-flash_1000",
        "summary_type_2": "response",
    },
    {
        "comparison": "v3_2flash_final_vs_v2",
        "summary_name_1": "summaries_v2_google-ip_gemini-2.0-flash_1000",
        "summary_type_1": "response_final",
        "summary_name_2": "summaries_v2_google-ip_gemini-2.0-flash_1000",
        "summary_type_2": "response_v2",
    },
    {
        "comparison": "v3_2flash_final_vs_v1",
        "summary_name_1": "summaries_v2_google-ip_gemini-2.0-flash_1000",
        "summary_type_1": "response_final",
        "summary_name_2": "summaries_v2_google-ip_gemini-2.0-flash_1000",
        "summary_type_2": "response",
    },
    # # # Comparando índices
    # {
    #     "comparison": "v3_2flash_final_google_vs_gte",
    #     "summary_name_1": "summaries_v2_google-ip_gemini-2.0-flash_1000",
    #     "summary_type_1": "response_final",
    #     "summary_name_2": "summaries_v2_gte-ip_gemini-2.0-flash_1000",
    #     "summary_type_2": "response_final",
    # },
    # # Comparando modelos
    {
        "comparison": "v4_2flash_vs_15flash_final",
        "summary_name_1": "summaries_v2_google-ip_gemini-2.0-flash_1000",
        "summary_type_1": "response_final",
        "summary_name_2": "summaries_v2_google-ip_gemini-1.5-flash-8b_1000",
        "summary_type_2": "response_final",
    },
    # {
    #     "comparison": "v3_2flash_vs_gemma_final",
    #     "summary_name_1": "summaries_v2_google-ip_gemini-2.0-flash_1000",
    #     "summary_type_1": "response_final",
    #     "summary_name_2": "summaries_v2_google-ip_gemma3_101",
    #     "summary_type_2": "response_final",
    # },
    # {
    #     "comparison": "v4_2flash_vs_gemma_final",
    #     "summary_name_1": "summaries_v2_google-ip_gemini-2.0-flash_1000",
    #     "summary_type_1": "response_final",
    #     "summary_name_2": "summaries_v2_google-ip_gemma3_101",
    #     "summary_type_2": "response_final",
    # },
]

EVALUATION_PROMPT_TEMPLATE = """

[Sistema]
Por favor, atue como um juiz imparcial e avalie a qualidade das respostas fornecidas por dois assistentes de IA para a pergunta do usuário exibida abaixo. Sua avaliação deve considerar a correção e a utilidade das respostas. Você receberá a resposta do Assistente A e a resposta do Assistente B. Seu trabalho é avaliar qual resposta é melhor.  

Primeiro, resolva a pergunta do usuário passo a passo de forma independente. Em seguida, compare as respostas dos dois assistentes com a sua. Identifique e corrija quaisquer erros.  

Evite qualquer viés de posição e assegure-se de que a ordem de apresentação das respostas não influencie sua decisão. Não permita que o tamanho das respostas influencie sua avaliação. Não favoreça certos nomes de assistentes. Seja o mais objetivo possível.  

Após fornecer uma explicação, apresente seu veredito final seguindo estritamente este formato: `"[[A]]"` se a resposta do Assistente A for melhor, `"[[B]]"` se a resposta do Assistente B for melhor, e `"[[C]]"` em caso de empate.

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


def select_summaries(df, comparison):
    dfi = (
        df[["nome", "summaries_name", "response", "response_v2", "response_final"]]
        .set_index("nome")
        .copy()
    )
    s1 = dfi[dfi.summaries_name == comparison["summary_name_1"]][
        comparison["summary_type_1"]
    ]
    s1.name = comparison["summary_name_1"] + "_" + comparison["summary_type_1"]
    s2 = dfi[dfi.summaries_name == comparison["summary_name_2"]][
        comparison["summary_type_2"]
    ]
    s2.name = comparison["summary_name_2"] + "_" + comparison["summary_type_2"]
    return s1, s2


def compare_summaries(
    df, comparisons, llm, limit=None, folder="data/evals", invert=True
):
    print("compare_summaries...")
    dfs = []
    for comparison in tqdm(comparisons, total=len(comparisons), desc="comparisons"):
        comparison_name = comparison["comparison"]
        print(f"{comparison_name=}")
        s1, s2 = select_summaries(df, comparison)
        if limit:
            s1 = s1.iloc[:limit]
            s2 = s2.iloc[:limit]
        evals = pairwise_eval_series(s1, s2, llm=llm)
        if invert:
            evals_ = pairwise_eval_series(s2, s1, llm=llm)
        df_evals = pd.DataFrame(evals)
        df_evals["pairwise_eval_"] = pd.DataFrame(evals_)["pairwise_eval"]
        df_evals["comparison"] = comparison_name
        out_path = f"{folder}/{comparison_name}.pq"
        df_evals.to_parquet(out_path)
        dfs.append(df_evals)
    return dfs


def load_summaries(summaries_folder):
    summaries_folder = Path(summaries_folder)
    files = os.listdir(summaries_folder)
    print(summaries_folder, len(files))
    hotels = []
    for file in files:
        dfi = pd.read_parquet(summaries_folder / file)
        hotels.append(dfi)
    df_results = pd.concat(hotels).reset_index(drop=True)
    return df_results


def load_prep_summaries(summaries=SUMMARIES):
    # Load
    dfs = []
    for s in summaries:
        df = load_summaries("data\\" + s)
        df["summary_name"] = s.replace("data\\", "")
        dfs.append(df)
    # Prep
    df_results = pd.concat(dfs)
    print(f"{df_results.shape=}")
    df_na = df_results[df_results.isna().any(axis=1)]
    print(f"{df_na.shape=}")
    df_results = df_results[~df_results.isna().any(axis=1)]
    print(f"{df_results.shape=}")
    n_sums = df_results.summary_name.nunique()
    print(f"{n_sums=}")
    hotel_counts = df_results.groupby("nome").count().iloc[:, 0]
    print(f"{len(hotel_counts)=}")
    hotels_common = hotel_counts[hotel_counts == n_sums].index
    print(f"{len(hotels_common)=}")
    df_results_common = df_results[df_results.nome.isin(hotels_common)].copy()
    print(f"{df_results_common.shape=}")
    return df_results_common


if __name__ == "__main__":
    if LOAD_PREP_SUMMARIES:
        df_summaries = load_prep_summaries()
        df_summaries.to_parquet("data/evals/df_summaries.pq")
    else:
        df_summaries = pd.read_parquet("data/evals/df_summaries.pq")

    llm, model_name, max_new_tokens = load_model(JUDGE_MODEL)

    dfs = compare_summaries(df_summaries, COMPARISONS, llm=llm, limit=LIMIT)
