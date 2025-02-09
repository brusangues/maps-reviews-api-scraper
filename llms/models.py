from datetime import datetime
import time
import os
import torch
import transformers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline

from analysis.src.utils import timeit

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# transformers.logging.set_verbosity_debug()
# logging.basicConfig(level=logging.DEBUG)
# set_debug(True)


models_text = {
    "llama1b": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "llama3b": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "gemma2b": "unsloth/gemma-2-2b-it-bnb-4bit",
    "phi": "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
}
models_embedding = {
    "gte": "Alibaba-NLP/gte-multilingual-base",
    "modernbert": "nomic-ai/modernbert-embed-base",
}
MAX_NEW_TOKENS = 100


@timeit
def load_model(model_alias="llama1b", max_new_tokens=MAX_NEW_TOKENS):
    print("load_model...")
    model_name = models_text.get(model_alias, models_text["llama1b"])
    max_new_tokens = min(int(max_new_tokens), MAX_NEW_TOKENS)
    print(f"{model_alias=} {model_name=}")
    pipe = transformers.pipeline(
        task="text-generation",
        # temperature=1e-10,
        # device=0,
        model=model_name,
        # pad_token_id=128001,
        max_new_tokens=max_new_tokens,
        return_full_text=False,
        # truncation=True, do_sample=True,
        # top_k=50, top_p=0.95,
    )
    # Instantiate the HuggingFacePipeline with the loaded pipeline
    llm = HuggingFacePipeline(pipeline=pipe)
    print(f"{llm=}")
    return llm, model_name


@timeit
def load_embedding(model_alias="gte"):
    print("load_embedding...")
    model_name = models_embedding.get(model_alias, models_embedding["gte"])
    print(f"{model_alias=} {model_name=}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs={"trust_remote_code": True}
    )
    print(f"{embeddings=}")
    return embeddings, model_name


@timeit
def query_model(llm: HuggingFacePipeline, prompt: str):
    print("query_model...")
    print(f"{len(prompt)=} {prompt=}")
    # Calculate the number of input tokens using the model tokenizer
    num_input_tokens = len(llm.pipeline.tokenizer.encode(prompt))
    print(f"{num_input_tokens=}")
    llmresult = llm.generate([prompt])
    response = llmresult.generations[0][0].text
    print(f"{len(response)=}")
    print(f"{response=}")
    return response


@timeit
async def query_model_async(llm: HuggingFacePipeline, prompt: str):
    print("\n", "=" * 20, "\n")
    print(f"prompt=\n{prompt}")
    print(f"\nresponse=")
    chunks = []
    start_time = time.time()

    # Calculate the number of input tokens using the model tokenizer
    num_input_tokens = len(llm.pipeline.tokenizer.encode(prompt))
    print(f"{num_input_tokens=}")

    async for chunk in llm.astream(prompt):
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    print("\n", "=" * 20, "\n")
    end_time = time.time()
    elapsed_time = end_time - start_time
    num_tokens = len(chunks)
    tokens_per_second = num_tokens / elapsed_time
    print(f"{num_tokens=} {tokens_per_second=} {elapsed_time=}")
    return "".join(chunks)


if __name__ == "__main__":
    llm, _ = load_model()
    query_model(llm, "How much is 1 + 1?")
