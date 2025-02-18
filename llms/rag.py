from uuid import uuid4
import faiss
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.globals import set_debug
import asyncio
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import logging
from datetime import datetime
import time
import json
from tqdm import tqdm
import os
import re

from llms.utils import timeit
from llms.models import load_model, load_embedding, query_model, query_model_async

# logging.basicConfig(level=logging.DEBUG)
# set_debug(True)

QUERY_PREFIX_E5_INSTRUCT = "Instruct: Given a Portuguese question, retrieve relevant hotel reviews that best answer the question. \nQuery: "
DOC_PREFIX = ""
QUERY_PREFIX = ""
PROMPT = """
Você é um assistente para tarefas de resposta a perguntas em português.
Leia a PERGUNTA e use os seguintes trechos de CONTEXTO recuperado de avaliações de hotéis para respondê-la.
Quanto menor o SCORE, maior é a similaridade da avaliação com a PERGUNTA.
Se você não souber a resposta, apenas diga que não sabe. 
Mantenha a resposta concisa.
Responda depois da tag RESPOSTA.
\n
PERGUNTA:\n{question}\n
CONTEXTO:\n{context}\n
RESPOSTA:"""

PROMPT_QUERY = """
Com base na PERGUNTA do usuário, monte um JSON de query de MongoDB com os seguintes operadores:
$eq (igual a)
$neq (não igual a)
$gt (maior que)
$lt (menor que)
$gte (maior ou igual a)
$lte (menor ou igual a)
$in (pertence à lista)
$nin (não pertence à lista)
$and (todas as condições devem corresponder)
$or (qualquer condição deve corresponder)
$not (negação da condição)
Evite de utilizar operadores $and e $or no nível superior de queries sem necessidade.

Os metadados nos quais esses operadores podem ser aplicados são os seguintes:
"nome": Nome do hotel.
"estrelas": Número de estrelas do hotel, variando de 0 a 5.
"cidade": Nome da cidade onde o hotel está localizado. Exemplos ["São Paulo", "Rio de Janeiro"]
"sigla_estado": Código de duas letras representando o estado brasileiro onde o hotel está localizado em caixa alta. ["RJ", "RO", "BA", "PE", "MG", "SP", "SC", "AM", "CE", "PR", "PA", "AL", "GO", "RS", "RN", "DF", "RR", "MS", "TO", "MT", "PB", "ES", "MA"]
"estado": Nome completo do estado onde o hotel está localizado em caixa alta. Exemplos ["SÃO PAULO", "RIO DE JANEIRO"]
"capital_estado": Nome da capital do estado onde o hotel está localizado em caixa alta. Exemplos ["SÃO PAULO", "RIO DE JANEIRO"]
"regiao": Região geográfica do Brasil onde o hotel está localizado. Valores possíveis: ["SUDESTE", "NORTE", "NORDESTE", "SUL", "CENTRO-OESTE"].
"endereco": Endereço completo do hotel, incluindo rua, número, bairro, cidade e estado.
"classificacao_geral": Nota geral do hotel baseada nas avaliações dos usuários, variando de 0.0 a 5.0.
"quantidade_avaliacoes": Número total de avaliações recebidas pelo hotel.
"nota_avaliacao": Nota específica da avaliação, variando de 1 a 5.
"curtidas_avaliacao": Número de curtidas recebidas por uma avaliação.
"usuario_guia_local": Indica se o usuário que fez a avaliação é um Guia Local do Google Maps (1 para sim, 0 para não).
"data_avaliacao": Data em que a avaliação foi feita, no formato "YYYY-MM-DD".
"nota_quartos": Nota dada pelos usuários para os quartos do hotel, variando de 1 a 5.
"nota_localizacao": Nota dada pelos usuários para a localização do hotel, variando de 1 a 5.
"nota_servico": Nota dada pelos usuários para o serviço do hotel, variando de 1 a 5.
"avaliacao_recente": Indica se a avaliação foi feita dentro de 6 meses (1 para sim, 0 para não).
"numero_palavras_avaliacao": Quantidade de palavras contidas na avaliação do usuário.

Após o JSON, inclua a pergunta reformulada, sem a parte que foi usada para montar a query.
Siga os seguintes exemplos:
PERGUNTA: Qual o melhor hotel com 3 estrelas ou mais?
RESPOSTA: ```json {"estrelas": {"$gte": 3}} ``` Qual o melhor hotel?\n
PERGUNTA: Qual o melhor hotel no estado de SP?
RESPOSTA: ```json {"sigla_estado": {"$eq": "SP"}} ``` Qual o melhor hotel?\n
PERGUNTA: Quero um hotel de 4 estrelas pé na areia.
RESPOSTA: ```json {"estrelas": {"$eq": 4}} ``` Quero um hotel pé na areia.\n
FIM DOS EXEMPLOS!!!\n
Não explique a resposta e responda apenas com o JSON seguido da pergunta reformulada.\n
"""
CREATE_INDEX = False
PATH_INDEX = "data/faiss_index_google_v4"
INDEX_BATCH_SIZE = 10_000
N_SAMPLES = None
N_RESPONSES = 10
MAX_VECTOR_STORES = 1000
PATH_DATA = "data/df_prep_2024-12-09_08-23-45_627733.pq"
PATH_DATA = "data/df_index_2025_v1.pq"
EMBEDDING_MODEL = "google-4"


@timeit
def load_data(n_samples=N_SAMPLES):
    print("Loading data...")
    df = pd.read_parquet(PATH_DATA)
    print(df.shape)
    if n_samples is not None and n_samples < len(df):
        df = df.sample(n_samples, random_state=42)
    print(df.shape)
    return df


@timeit
def create_index(df, embeddings: HuggingFaceEmbeddings):
    print("Creating index...")

    # Loading docs
    print("Loading docs...")
    docs = []
    for i, (id, row) in enumerate(df.iterrows()):
        # print(i, id, row["name"])
        # print(row.text)
        metadata = row.to_dict()
        text = metadata.pop("text")
        text = DOC_PREFIX + text
        # print(metadata)
        doc = Document(page_content=text, metadata=metadata, id=str(uuid4()))
        docs.append(doc)

    # Embedding docs
    print("Embedding docs...")
    len_docs = len(docs)
    batch_size = INDEX_BATCH_SIZE
    for i in tqdm(range(0, len_docs, batch_size)):
        batch = docs[i : i + batch_size]
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hi man")))
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        vector_store.add_documents(documents=batch)
        vector_store.save_local(PATH_INDEX + f"/{i}")
    print("Index creation finished!")


@timeit
def load_index(embeddings: HuggingFaceEmbeddings):
    print("Loading index...")
    # vector_store = FAISS.load_local(
    #     PATH_INDEX, embeddings, allow_dangerous_deserialization=True
    # )

    folders = os.listdir(PATH_INDEX)

    vector_store = None
    i = 0
    for f in tqdm(folders, total=len(folders)):
        if i >= MAX_VECTOR_STORES:
            break
        if vector_store is None:
            vector_store = FAISS.load_local(
                f"{PATH_INDEX}/{f}",
                embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            vector_i = FAISS.load_local(
                f"{PATH_INDEX}/{f}",
                embeddings,
                allow_dangerous_deserialization=True,
            )
            vector_store.merge_from(vector_i)
        print(f"{i=} {vector_store.index.ntotal=}")
        i += 1
    print("Index loaded.")
    return vector_store


@timeit
def query_index(vector_store: FAISS, query, filter: dict = {}):
    print("Querying index...")
    query = QUERY_PREFIX + query
    results = vector_store.similarity_search_with_score(
        query,
        k=N_RESPONSES,
        filter=filter,
    )
    context = ""
    for i, (res, score) in enumerate(results):
        print_ = (
            f" - Avaliação {i+1}, Similaridade: {score:0.3f}\n"
            f"Hotel: {res.metadata['nome']}, {res.metadata['estrelas']} Estrelas. "
            f"Região:{res.metadata['regiao']}; Estado:{res.metadata['sigla_estado']}\n"
            f"Nota: {res.metadata['nota_avaliacao']}\nAvaliação: {res.page_content}\n\n"
        )
        print(res.metadata, "\n", print_)
        context += print_
    return results, context


def query_with_user_input(vector_store: FAISS, llm: HuggingFacePipeline):
    print("Starting main loop...")
    print("=" * 100)
    while True:
        query = input("\nEnter query:\n")
        if "exit" in query:
            print("Exiting...")
            break

        filter_raw = input("\nEnter filter json:\n")
        filter = {}
        if filter_raw != "":
            try:
                filter = json.loads(filter_raw)
                print(f"filter parsed:\n{filter}")
            except Exception as e:
                print(e)

        results, context = query_index(vector_store, query, filter)
        prompt = PROMPT.format(
            context=context,
            question=query,
        )
        # response = asyncio.run(query_model_async(llm, prompt))
        response = query_model(llm, prompt)
        # print(response)
        print("\n", "=" * 100, "\n")


def query_make_filter(llm: HuggingFacePipeline, query: str):
    prompt_query = PROMPT_QUERY + f"PERGUNTA: {query}\nRESPOSTA: "
    filter_raw = query_model(llm, prompt_query)
    filter = {}
    query_updated = None
    try:
        filter_query = filter_raw.replace("```json", "").split("```")
        query_updated = filter_query[1].strip()
        filter_raw = filter_query[0].replace("'", '"')
        filter = json.loads(filter_raw)
        print(f"Filtro carregado via llm:\n{filter}")
    except Exception as e:
        print(e)
    query_updated = query if query_updated is None else query_updated
    print(f"{filter=}\n{query_updated=}")
    return filter, query_updated


def rag_loop(vector_store: FAISS, llm: HuggingFacePipeline, query: str):
    filter, query_updated = query_make_filter(llm, query)
    results, context = query_index(vector_store, query_updated, filter)
    prompt = PROMPT.format(
        context=context,
        question=query,
    )
    response = query_model(llm, prompt)
    return response


def query_with_user_input_v2(vector_store: FAISS, llm: HuggingFacePipeline):
    print("Starting main loop...")
    print("=" * 100)
    while True:
        query = input("\nEnter query:\n")
        if "exit" in query:
            print("Exiting...")
            break

        rag_loop(vector_store, llm, query)
        print("\n", "=" * 100, "\n")


if __name__ == "__main__":
    if CREATE_INDEX:
        embeddings, _ = load_embedding(EMBEDDING_MODEL, task_type="retrieval_document")
        df_sample = load_data(N_SAMPLES)
        create_index(df_sample, embeddings)
    else:
        embeddings, _ = load_embedding(EMBEDDING_MODEL, task_type="retrieval_query")
        vector_store = load_index(embeddings)

        llm, _, _ = load_model("gemini-2.0-flash")

        query_with_user_input_v2(vector_store, llm)
