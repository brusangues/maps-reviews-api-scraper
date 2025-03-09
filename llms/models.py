from datetime import datetime
import time
import os
import torch
import transformers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.outputs import Generation, LLMResult

from analysis.src.utils import timeit

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MAX_NEW_TOKENS = 3000


# fmt: off
models_text = {
    "llama1b": ("local", "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"),
    "llama3b": ("local", "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"),
    "gemma2b": ("local", "unsloth/gemma-2-2b-it-bnb-4bit"),
    "phi":     ("local", "unsloth/Phi-3.5-mini-instruct-bnb-4bit"),

    "gemini-2.0-flash":      ("google", "gemini-2.0-flash"),
    "gemini-2.0-flash-lite": ("google", "gemini-2.0-flash-lite"),
    "gemini-1.5-flash":      ("google", "gemini-1.5-flash"),
    "gemini-1.5-flash-8b":   ("google", "gemini-1.5-flash-8b"),
    "gemini-1.5-pro":        ("google", "gemini-1.5-pro"),
    "gemini-2.0-pro":        ("google", "gemini-2.0-pro-exp"),
    "gemini-2.0-flash-thinking": ("google", "gemini-2.0-flash-thinking-exp"),

    # "gemini-2.0-pro": ("open_router", "google/gemini-2.0-pro-exp-02-05:free", "https://openrouter.ai/api/v1"),
    # "gemini-2.0-flash-thinking": ("open_router", "google/gemini-2.0-flash-thinking-exp:free", "https://openrouter.ai/api/v1"),
    "or-llama70b":    ("open_router", "meta-llama/llama-3.3-70b-instruct:free", "https://openrouter.ai/api/v1"),
    "or-r1":          ("open_router", "deepseek/deepseek-r1:free", "https://openrouter.ai/api/v1"),
    "qwen2.5":        ("open_router", "qwen/qwen2.5-vl-72b-instruct:free", "https://openrouter.ai/api/v1"),

    "gpt-4o":      ("github", "gpt-4o", "https://models.inference.ai.azure.com"),
    "gpt-4o-mini": ("github", "gpt-4o-mini", "https://models.inference.ai.azure.com"),
    "gh-r1":       ("github", "DeepSeek-R1", "https://models.inference.ai.azure.com"),
    "gh-llama70b": ("github", "Llama-3.3-70B-Instruct", "https://models.inference.ai.azure.com"),
}
models_embedding = {
    "gte":         ("local", "Alibaba-NLP/gte-multilingual-base"),
    "modernbert":  ("local", "nomic-ai/modernbert-embed-base"),
    "e5":          ("local", "intfloat/multilingual-e5-large"),
    "e5-instruct": ("local", "intfloat/multilingual-e5-large-instruct"),
    "arctic":      ("local", "Snowflake/snowflake-arctic-embed-l-v2.0"),
    "google-4":    ("google", "models/text-embedding-004"),
    "gemini":      ("google", "models/gemini-embedding-exp-03-07"),
}
# fmt: on
MAX_NEW_TOKENS = 3000


class GenericHuggingFacePipeline:
    def __init__(self, model):
        self.model = model
        self.pipeline = None

    def generate(self, prompts, **kwargs):
        results = []
        for prompt in prompts:
            response = self.model.invoke(prompt)
            metadata = {}
            try:
                metadata = response.usage_metadata
                metadata = metadata | response.response_metadata
            except:
                pass
            generation = Generation(text=response.content, generation_info=metadata)
            results.append([generation])
        return LLMResult(generations=results)


@timeit
def load_model(model_alias="gemini-2.0-flash", max_new_tokens=MAX_NEW_TOKENS):
    print("load_model...")
    model_data = models_text.get(model_alias, models_text["gemini-2.0-flash"])
    print(f"{model_alias=} {max_new_tokens=} {model_data=}")

    provider = model_data[0]
    model_name = model_data[1]

    if provider in ["open_router", "github"]:
        print("Loading OpenRouter or Github model...")
        open_router_key = os.environ["OPEN_ROUTER_API_KEY"]
        github_key = os.environ["GITHUB_TOKEN"]
        api_key = github_key if provider == "github" else open_router_key
        base_url = model_data[2]
        llm = ChatOpenAI(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            # max_new_tokens=max_new_tokens,
            max_completion_tokens=max_new_tokens,
        )
        print(f"{llm=}")
        hf_pipe = GenericHuggingFacePipeline(model=llm)
    elif provider == "google":
        print("Loading ChatGoogleGenerativeAI model...")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_output_tokens=max_new_tokens,
            timeout=10,
            max_retries=2,
        )
        print(f"{llm=}")
        hf_pipe = GenericHuggingFacePipeline(model=llm)
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
        hf_pipe = HuggingFacePipeline(pipeline=pipe)
    print(f"{hf_pipe=}")
    return hf_pipe, model_name, max_new_tokens


@timeit
def load_embedding(model_alias="google-4", task_type="retrieval_query"):
    print("load_embedding...")
    model_data = models_embedding.get(model_alias, models_embedding["google-4"])
    print(f"{model_alias=} {model_data=} {task_type=}")
    provider = model_data[0]
    model_name = model_data[1]

    if provider == "google":
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
    print(f"{len(prompt)=}")
    # print(f"{len(prompt)=} {prompt=}")
    # Calculate the number of input tokens using the model tokenizer
    if llm.pipeline is None:
        num_input_tokens = len(prompt.split())
    else:
        num_input_tokens = len(llm.pipeline.tokenizer.encode(prompt))
    print(f"{num_input_tokens=}")
    llmresult = llm.generate([prompt])
    response = llmresult.generations[0][0].text
    info = llmresult.generations[0][0].generation_info
    print(f"{len(response)=}")
    print(f"{response=}")
    print(f"{info=}")
    return response, info


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
