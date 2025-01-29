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

from analysis.src.utils import timeit

# logging.basicConfig(level=logging.DEBUG)
# set_debug(True)

# MODEL = "nomic-ai/modernbert-embed-base"
# DOC_PREFIX = "search_document: "
# QUERY_PREFIX = "search_query: "

EMBEDDING_MODEL = "Alibaba-NLP/gte-multilingual-base"
LLM_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
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
N_SAMPLES = 1000


@timeit
def load_data(n_samples):
    print("Loading data...")
    df = pd.read_parquet("data/df_prep_2024-12-09_08-23-45_627733.pq")
    df = df.query("~text.isna() & text.str.len() > 20")[
        ["name", "stars", "region", "state", "rating", "text"]
    ]
    df = df.sample(n_samples, random_state=42)
    print(df.shape)
    return df


@timeit
def create_index(df, embeddings):
    print("Creating index...")
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hi man")))
    print(f"{index=}")
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    print(f"empty {vector_store=}")

    # Loading docs
    print("Loading docs...")
    docs = []
    for i, (id, row) in enumerate(df.iterrows()):
        print(i, id, row["name"])
        # print(row.text)
        metadata = row.to_dict()
        text = metadata.pop("text")
        text = DOC_PREFIX + text
        # print(metadata)
        doc = Document(page_content=text, metadata=metadata, id=str(uuid4()))
        docs.append(doc)

    # Embedding docs
    print("Embedding docs...")
    vector_store.add_documents(documents=docs)

    # Saving index
    print("Saving index...")
    vector_store.save_local("data/faiss_index")
    return vector_store


@timeit
def load_index(embeddings):
    print("Loading index...")
    vector_store = FAISS.load_local(
        "data/faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    print("Index loaded.")
    return vector_store


@timeit
def query_index(vector_store, query):
    print("Querying index...")
    query = QUERY_PREFIX + query
    results = vector_store.similarity_search_with_score(
        query,
        k=4,
        # filter={"source": "tweet"},
    )
    context = ""
    for i, (res, score) in enumerate(results):
        print_ = (
            f" - Avaliação {i+1}:\n"
            f"Hotel: {res.metadata['name']}, {res.metadata['stars']} Estrelas. "
            # f"Região:{res.metadata['region']}; Estado:{res.metadata['state']}\n"
            f"Nota: {res.metadata['rating']}\nComentário: {res.page_content}\n\n"
        )
        print(score, print_)
        context += print_
    return results, context


def query_with_user_input(vector_store, llm):
    print("Starting main loop...")
    print("=" * 100)
    while True:
        query = input("\nEnter query:\n")
        if "exit" in query:
            print("Exiting...")
            break
        results, context = query_index(vector_store, query)
        prompt = PROMPT.format(
            context=context,
            question=query,
        )
        response = asyncio.run(async_generate(llm, prompt))
        # print(response)
        print("\n", "=" * 100, "\n")


@timeit
def load_model():
    print("load_model...")
    pipe = pipeline(
        task="text-generation",
        temperature=1e-5,
        # device=0,
        model=LLM_MODEL,
        pad_token_id=128001,
        max_new_tokens=100,
        return_full_text=False,
        # truncation=True, do_sample=True,
        # top_k=50, top_p=0.95,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


@timeit
def load_embedding_model():
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"trust_remote_code": True}
    )
    print(f"{embeddings=}")
    return embeddings


async def async_generate(llm: HuggingFacePipeline, prompt):
    print(f"{datetime.now()=}")
    print("\n", "=" * 20, "\n")
    print(f"prompt=\n{prompt}")
    print(f"\nresponse=")
    chunks = []
    start_time = time.time()
    async for chunk in llm.astream(prompt):
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    print("\n", "=" * 20, "\n")
    end_time = time.time()
    elapsed_time = end_time - start_time
    num_tokens = len(chunks)
    tokens_per_second = num_tokens / elapsed_time
    print(f"{elapsed_time=}")
    print(f"{tokens_per_second=}")
    print(f"{datetime.now()=}")
    return "".join(chunks)


if __name__ == "__main__":
    print("Loading data...")
    df_sample = load_data(N_SAMPLES)
    embeddings = load_embedding_model()
    if CREATE_INDEX:
        vector_store = create_index(df_sample, embeddings)
    else:
        vector_store = load_index(embeddings)

    llm = load_model()

    query_with_user_input(vector_store, llm)
