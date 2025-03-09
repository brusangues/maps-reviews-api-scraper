import time
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from llms.summary import (
    prep_data,
    make_prompt_summary_full,
    make_prompt_summary_full_v2,
    make_prompts_summary_topic,
    make_prompt_final,
    N_RESPONSES_TOPIC,
    N_RESPONSES_FULL,
)
from llms.models import load_model, query_model
from llms.g_eval import g_eval_scores
from llms.rag import load_data, load_rag

SUMMARIES_FOLDER = "data/hotel_summaries"


def run_make_summaries(
    df: pd.DataFrame,
    hotel_counts: pd.DataFrame,
    vector_store,
    rag_alias: str,
    llm,
    model_name: str,
    max_new_tokens: int,
) -> list:
    print("run_make_summaries...")
    results_summaries = []
    for i, hotel in tqdm(hotel_counts.iterrows(), total=len(hotel_counts)):
        hotel_name = hotel.nome
        count_ = hotel.count_
        n_responses_full = min(count_, N_RESPONSES_FULL)
        n_responses_topic = min(count_, N_RESPONSES_TOPIC)
        print("\n", "=" * 100)
        print(f"{i=} {hotel_name=} {count_=} {n_responses_full=} {n_responses_topic=}")

        prompt, context = make_prompt_summary_full(hotel_name, df)
        response, info = query_model(llm, prompt)

        prompt_v2, context_v2 = make_prompt_summary_full_v2(
            hotel_name=hotel_name,
            vector_store=vector_store,
            n_responses=n_responses_full,
        )
        response_v2, info_v2 = query_model(llm, prompt_v2)

        prompts, contexts = make_prompts_summary_topic(
            hotel_name=hotel_name,
            vector_store=vector_store,
            n_responses=n_responses_topic,
        )
        responses = []
        infos = []
        for p in prompts:
            r, i = query_model(llm, p)
            responses.append(r)
            infos.append(i)

        prompt_final, context_final = make_prompt_final(
            hotel_name=hotel_name,
            responses=responses,
        )
        response_final, info_final = query_model(llm, prompt_final)
        results_summaries.append(
            {
                "model_name": model_name,
                "max_new_tokens": max_new_tokens,
                "rag_alias": rag_alias,
                "prompt": prompt,
                "response": response,
                "prompt_v2": prompt_v2,
                "response_v2": response_v2,
                "prompts": prompts,
                "responses": responses,
                "prompt_final": prompt_final,
                "response_final": response_final,
                "context": context,
                "context_v2": context_v2,
                "contexts": contexts,
                "context_final": context_final,
                "info": info,
                "info_v2": info_v2,
                "infos": infos,
                "info_final": info_final,
            }
        )
    return results_summaries


def run_eval_summaries(
    hotel_counts: pd.DataFrame,
    llm,
    model_name: str,
    max_new_tokens: int,
) -> list:
    print("run_eval_summaries...")
    results_scores = []
    for i, hotel in tqdm(hotel_counts.iterrows(), total=len(hotel_counts)):
        hotel_name = hotel.nome
        print("\n", "=" * 100)
        print(f"{i=} {hotel_name=}")
        scores, scores_mean = g_eval_scores(
            [hotel.response], hotel.context, llm, 2  # noqa
        )
        scores_v2, scores_mean_v2 = g_eval_scores(
            [hotel.response_v2], hotel.context_v2, llm, 2
        )
        scores_final, scores_mean_final = g_eval_scores(
            [hotel.response_final], hotel.context_final, llm, 2
        )

        results_scores.append(
            {
                "eval_model_name": model_name,
                "eval_max_new_tokens": max_new_tokens,
                "scores": scores,
                "scores_mean": scores_mean,
                "scores_v2": scores_v2,
                "scores_mean_v2": scores_mean_v2,
                "scores_final": scores_final,
                "scores_mean_final": scores_mean_final,
            }
        )
    return results_scores


def choose_hotel(hotel_counts):
    # Escolhendo hotel que ainda não tenha resumos
    files = os.listdir(SUMMARIES_FOLDER)
    hotels_done = [f.replace(".pq", "") for f in files]
    hotels_left = hotel_counts[~hotel_counts.nome.isin(hotels_done)]
    print(f"{len(hotels_left)=}")
    if len(hotels_left) == 0:
        raise Exception("Todos os hotéis tem resumo")
    hotel_counts_sample = hotels_left.iloc[0:1].copy().reset_index(drop=True)
    print(hotel_counts_sample)
    return hotel_counts_sample


def run_make_summary_eval_hotel(
    df: pd.DataFrame,
    hotel_counts: pd.DataFrame,
    vector_store,
    rag_alias: str,
    llm,
    model_name: str,
    max_new_tokens: int,
):
    # Sumários
    results_summaries = run_make_summaries(
        df=df,
        hotel_counts=hotel_counts,
        vector_store=vector_store,
        rag_alias=rag_alias,
        llm=llm,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
    )
    df_results = pd.DataFrame(results_summaries)
    hotel_counts[list(results_summaries[0])] = df_results
    print(hotel_counts.iloc[0])

    # Avaliações
    time.sleep(10)
    try:
        results_scores = run_eval_summaries(
            hotel_counts=hotel_counts,
            llm=llm,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
        )
        df_scores = pd.DataFrame(results_scores)
        hotel_counts[list(results_scores[0])] = df_scores
        print(hotel_counts.iloc[0])
    except Exception as e:
        print(e)

    # Salvando resultado
    hotel = hotel_counts.iloc[0].nome
    out_path = f"{SUMMARIES_FOLDER}/{hotel}.pq"
    hotel_counts.to_parquet(out_path)
    print(out_path)


def main():
    # Carregando dados, modelos e rag
    df = load_data()
    hotel_counts = prep_data(df)
    vector_store, rag_alias, embeddings_name = load_rag("google-ip")
    llm, model_name, max_new_tokens = load_model("gemini-2.0-flash")

    # Garantindo que pasta existe
    summaries_folder = Path(SUMMARIES_FOLDER)
    summaries_folder.mkdir(parents=True, exist_ok=True)

    max_len = len(hotel_counts)
    for i in tqdm(range(max_len), total=max_len):
        print(f"\n{i=}", "_" * 200)
        hotel_counts_sample = choose_hotel(hotel_counts)
        run_make_summary_eval_hotel(
            df=df,
            hotel_counts=hotel_counts_sample,
            vector_store=vector_store,
            rag_alias=rag_alias,
            llm=llm,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
        )


if __name__ == "__main__":
    main()
