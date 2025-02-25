from uuid import uuid4
import faiss
import pandas as pd
import numpy as np
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
from llms.prompts import PROMPT, PROMPT_QUERY, format_context

# logging.basicConfig(level=logging.DEBUG)
# set_debug(True)

QUERY_PREFIX_E5_INSTRUCT = "Instruct: Given a Portuguese question, retrieve relevant hotel reviews that best answer the question. \nQuery: "
DOC_PREFIX = ""
QUERY_PREFIX = ""
CREATE_INDEX = False
NORMALIZE_INDEX = False
PATH_INDEX = "data/faiss_index_gte_v5"
INDEX_BATCH_SIZE = 10_000
N_SAMPLES = None
N_RESPONSES = 5
FETCH_K = 1000
MAX_VECTOR_STORES = 1000
# PATH_DATA = "data/df_prep_2024-12-09_08-23-45_627733.pq"
PATH_DATA = "data/df_index_2025_v1.pq"
EMBEDDING_MODEL = "gte"
RAG_ALIAS = "google-ip"
INDEX_SIZE = 222923

rags = {
    "gte-old": ["gte", "data/faiss_index_full_v2"],
    "gte-l2": ["gte", "data/faiss_index_gte_v5"],
    "gte-ip": ["gte", "data/faiss_index_gte_v5_ip"],
    "google-l2": ["google-4", "data/faiss_index_google_v4"],
    "google-ip": ["google-4", "data/faiss_index_google_v4_ip"],
}


def load_rag(rag_alias: str):
    print("load_rag...")
    rag_info = rags.get(rag_alias, rags["google-ip"])
    embedding_model_alias = rag_info[0]
    path_index = rag_info[1]
    embeddings, embeddings_name = load_embedding(
        embedding_model_alias, task_type="retrieval_query"
    )
    print(f"{rag_alias=} {embedding_model_alias=} {embeddings_name=} {path_index=}")
    vector_store = load_index(embeddings, path_index)
    return vector_store, rag_alias, embeddings_name


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
def create_index(
    df,
    embeddings: HuggingFaceEmbeddings,
    index_type: str = "ip",
    path_index: str = PATH_INDEX,
):
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
        d = len(embeddings.embed_query("hi man"))
        if index_type == "ip":
            index = faiss.IndexFlatIP(d)
        else:
            index = faiss.IndexFlatL2(d)
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        vector_store.add_documents(documents=batch)
        vector_store.save_local(path_index + f"/{i}")
    print("Index creation finished!")


# Function to normalize embeddings
def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@timeit
def normalize_index(embeddings: HuggingFaceEmbeddings, path_index: str = PATH_INDEX):
    print("Normalizing index...")
    # Os embeddings da google e do gte já são normalizados
    folders = os.listdir(path_index)

    vector_store = None
    i = 0
    for f in tqdm(folders, total=len(folders)):
        print(f"{i=}")
        if i >= MAX_VECTOR_STORES:
            break
        vector_store_l2 = FAISS.load_local(
            f"{path_index}/{f}",
            embeddings,
            allow_dangerous_deserialization=True,
        )

        index: faiss.Index = vector_store_l2.index
        vectors = index.reconstruct_n(0, index.ntotal)

        index_ip = faiss.IndexFlatIP(index.d)
        index_ip.add(vectors)

        vector_store_ip = FAISS(
            embedding_function=embeddings,
            index=index_ip,
            docstore=InMemoryDocstore(vector_store_l2.docstore._dict),
            index_to_docstore_id=vector_store_l2.index_to_docstore_id,
        )

        assert vector_store_ip.index.d == vector_store_l2.index.d
        assert vector_store_ip.index.ntotal == vector_store_l2.index.ntotal
        assert vector_store_ip.docstore._dict == vector_store_l2.docstore._dict
        assert (
            vector_store_ip.index_to_docstore_id == vector_store_l2.index_to_docstore_id
        )
        assert (
            vector_store_ip.index.reconstruct_n(0, 1)[0]
            == vector_store_l2.index.reconstruct_n(0, 1)[0]
        ).all()
        norm_ = np.linalg.norm(vector_store_ip.index.reconstruct_n(0, 1)[0])
        assert abs(norm_ - 1) < 1e-4
        vector_store_ip.save_local(path_index + f"_ip/{i}")
        i += 1
    print("Index normalized.")


@timeit
def load_index(embeddings: HuggingFaceEmbeddings, path_index: str = PATH_INDEX):
    print("Loading index...")
    folders = os.listdir(path_index)

    vector_store = None
    i = 0
    for f in tqdm(folders, total=len(folders)):
        if i >= MAX_VECTOR_STORES:
            break
        if vector_store is None:
            vector_store = FAISS.load_local(
                f"{path_index}/{f}",
                embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            vector_i = FAISS.load_local(
                f"{path_index}/{f}",
                embeddings,
                allow_dangerous_deserialization=True,
            )
            vector_store.merge_from(vector_i)
        print(f"{i=} {vector_store.index.ntotal=}")
        i += 1
    print("Index loaded.")
    return vector_store


@timeit
def query_index(
    vector_store: FAISS, query, filter: dict = {}, n_responses: int = N_RESPONSES
):
    print("Querying index...")
    query = QUERY_PREFIX + query
    i = 0
    while True:
        fetch_k = FETCH_K * (10**i)
        fetch_k = min(fetch_k, INDEX_SIZE)
        print(f"Query with increasing fetch_k: {i=} {fetch_k=}")
        results = vector_store.similarity_search_with_score(
            query,
            k=fetch_k,
            fetch_k=fetch_k,
            filter=filter,
        )
        print(f"{len(results)=}/{fetch_k=} {n_responses=}")
        if len(results) >= n_responses:
            print("Results sufficient.")
            break
        elif fetch_k >= INDEX_SIZE or i >= 3:
            print("Max queries reached!")
            break
        i += 1

    # Construindo contexto
    results = sorted(results, key=lambda x: x[1])[:n_responses]
    context = ""
    for i, (res, score) in enumerate(results):
        context_i = format_context(i, res, score)
        # print(res.metadata, "\n", context_i)
        print(context_i)
        context += context_i
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
    elif NORMALIZE_INDEX:
        embeddings, _ = load_embedding(EMBEDDING_MODEL, task_type="retrieval_document")
        normalize_index(embeddings)
    else:
        # embeddings, _ = load_embedding(EMBEDDING_MODEL, task_type="retrieval_query")
        # vector_store = load_index(embeddings)
        vector_store, _, _ = load_rag(RAG_ALIAS)

        llm, _, _ = load_model("gemini-2.0-flash")

        query_with_user_input_v2(vector_store, llm)
