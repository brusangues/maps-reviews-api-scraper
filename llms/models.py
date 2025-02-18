from datetime import datetime
import time
import os
import torch
import transformers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.outputs import Generation, LLMResult

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
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.0-flash-lite": "gemini-2.0-flash-lite-preview-02-05",
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-1.5-flash-8b": "gemini-1.5-flash-8b",
    "gemini-1.5-pro": "gemini-1.5-pro",
}
models_embedding = {
    "gte": "Alibaba-NLP/gte-multilingual-base",
    "modernbert": "nomic-ai/modernbert-embed-base",
    "e5": "intfloat/multilingual-e5-large",
    "e5-instruct": "intfloat/multilingual-e5-large-instruct",
    "arctic": "Snowflake/snowflake-arctic-embed-l-v2.0",
    "google-4": "models/text-embedding-004",
    # "google-5": "models/text-embedding-005",
    # "google-m2": "models/text-multilingual-embedding-002",
}
MAX_NEW_TOKENS = 3000


class GeminiHuggingFacePipeline:
    def __init__(self, model):
        self.model = model
        self.pipeline = None

    def generate(self, prompts, **kwargs):
        results = []
        for prompt in prompts:
            response = self.model.invoke(prompt)
            generation = Generation(text=response.content)
            results.append([generation])
        return LLMResult(generations=results)


@timeit
def load_model(model_alias="gemini-2.0-flash", max_new_tokens=MAX_NEW_TOKENS):
    print("load_model...")
    model_name = models_text.get(model_alias, models_text["gemini-2.0-flash"])
    print(f"{model_alias=} {model_name=} {max_new_tokens=}")
    if model_name.startswith("gemini"):
        print("Loading ChatGoogleGenerativeAI model...")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_new_tokens=max_new_tokens,
            timeout=10,
            max_retries=2,
        )
        llm = GeminiHuggingFacePipeline(model=llm)
    else:
        print("Loading transformers.pipeline local model...")
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
        llm = HuggingFacePipeline(pipeline=pipe)
    print(f"{llm=}")
    return llm, model_name, max_new_tokens


@timeit
def load_embedding(model_alias="google-4", task_type="retrieval_query"):
    print("load_embedding...")
    model_name = models_embedding.get(model_alias, models_embedding["google-4"])
    print(f"{model_alias=} {model_name=} {task_type=}")
    if model_alias.startswith("google"):
        print("Loading GoogleGenerativeAIEmbeddings model...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            task_type=task_type,
        )
    else:
        print("Loading HuggingFaceEmbeddings model...")
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
    if llm.pipeline is None:
        num_input_tokens = len(prompt.split())
    else:
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
    llm, _, _ = load_model()
    query_model(llm, "How much is 1 + 1?")
