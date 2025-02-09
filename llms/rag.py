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

from llms.utils import timeit
from llms.models import load_model, load_embedding, query_model_async

# logging.basicConfig(level=logging.DEBUG)
# set_debug(True)

DOC_PREFIX = ""
QUERY_PREFIX = ""
PROMPT = """
Você é um assistente para tarefas de resposta a perguntas em português.
Leia a PERGUNTA e use os seguintes trechos de CONTEXTO recuperado para respondê-la.
Se você não souber a resposta, apenas diga que não sabe. 
Use apenas uma frase e mantenha a resposta concisa, em poucas palavras.
Responda depois da tag RESPOSTA.
\n
PERGUNTA:\n{question}\n
CONTEXTO:\n{context}\n
RESPOSTA:"""
CREATE_INDEX = False
PATH_INDEX = "data/faiss_index_full_v2"
INDEX_BATCH_SIZE = 20_000
N_SAMPLES = None
N_RESPONSES = 5
MAX_VECTOR_STORES = 1000


@timeit
def load_data(n_samples=N_SAMPLES):
    print("Loading data...")
    df = pd.read_parquet("data/df_prep_2024-12-09_08-23-45_627733.pq")
    print(df.shape)
    df = df.query("~text.isna() & text.str.len() > 20")[
        ["name", "stars", "region", "state", "rating", "text"]
    ]
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
            f" - Avaliação {i+1}, Score: {score:0.3f}\n"
            f"Hotel: {res.metadata['name']}, {res.metadata['stars']} Estrelas. "
            # f"Região:{res.metadata['region']}; Estado:{res.metadata['state']}\n"
            f"Nota: {res.metadata['rating']}\nComentário: {res.page_content}\n\n"
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
        response = asyncio.run(query_model_async(llm, prompt))
        # print(response)
        print("\n", "=" * 100, "\n")


if __name__ == "__main__":
    embeddings, _ = load_embedding()
    if CREATE_INDEX:
        df_sample = load_data(N_SAMPLES)
        create_index(df_sample, embeddings)

    vector_store = load_index(embeddings)

    llm, _ = load_model()

    query_with_user_input(vector_store, llm)
