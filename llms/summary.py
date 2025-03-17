import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from llms.rag import load_data, query_index
from llms.prompts import format_context

PROMPT_SUMMARY = """
Você é um assistente de sumarização de hotéis em português.
Utilize os seguintes trechos de CONTEXTO recuperado de avaliações de hotéis para escrever o RESUMO.
Siga as INSTRUÇÕES do usuário para escrever um resumo detalhado do que se pede.\n
CONTEXTO:\n{context}\n
INSTRUÇÕES:\n{query}\n
RESUMO:"""
TOPICS = [
    "Infraestrutura e Acomodações – Conforto, limpeza, tecnologia, lazer, estacionamento.",
    "Atendimento e Serviço – Cordialidade, eficiência, limpeza, concierge, check-in ágil.",
    "Localização e Acessibilidade – Proximidade, transporte, segurança, acessibilidade.",
    "Alimentação e Bebidas – Café da manhã, restaurante, serviço de quarto, qualidade.",
    "Experiência e Entretenimento – Lazer, eventos, recreação, passeios, parcerias.",
    "Custo-benefício e Políticas – Preço justo, flexibilidade, transparência, fidelidade.",
]
MAX_CONTEXT_LEN = None  # 50_000
# og settings
N_RESPONSES_TOPIC = 100
N_RESPONSES_FULL = 1_000
N_DOCS_MAX = 1_000

# newer settings
# N_RESPONSES_TOPIC = 1_000
# N_RESPONSES_FULL = 2_000
# N_DOCS_MAX = 2_000


def prep_data(df: pd.DataFrame) -> pd.DataFrame:
    hotel_counts = (
        df.groupby("nome")
        .agg(
            count_=("text", "count"),
        )
        .sort_values("count_", ascending=False)
        .reset_index()
    )

    hotel_counts = hotel_counts.query("count_>=10").copy()
    hotel_counts.loc[:, "big"] = hotel_counts.count_ >= 1000

    return hotel_counts


def make_docs(
    df_hotel: pd.DataFrame, n_docs: int = N_DOCS_MAX
) -> list[tuple[Document, None]]:
    print("make_docs...")
    df_hotel = df_hotel.sort_values("data_avaliacao", ascending=False)
    df_hotel = df_hotel.head(n_docs)
    docs = []
    for i, (id, row) in enumerate(df_hotel.iterrows()):
        metadata = row.to_dict()
        text = metadata.pop("text")
        doc = Document(page_content=text, metadata=metadata)
        docs.append((doc, None))
    print(f"{len(docs)=}")
    return docs


def make_context_summary(docs) -> str:
    n_docs = len(docs)
    meta = docs[0][0].__dict__["metadata"]
    context = (
        "Informações gerais sobre o hotel:"
        f"Hotel: {meta['nome']}, {int(meta['estrelas'])} Estrelas.\n"
        f"Região:{meta['regiao']}; Estado:{meta['estado']}; Cidade:{meta['cidade']}\n"
        f"Tipo:{meta['subcategoria']}; Classificação:{meta['classificacao_geral']}; Quantidade Avaliações:{int(meta['quantidade_avaliacoes'])}\n"
        "\n"
        f"Top {n_docs} avaliações do hotel mais semelhantes à pergunta:\n"
    )

    for i, (doc, score) in enumerate(docs):
        context_i = format_context(i, doc, score, include_hotel_context=False)
        context += context_i

    if MAX_CONTEXT_LEN is not None:
        context = context[:MAX_CONTEXT_LEN]
    return context


def make_context_final(hotel: str, responses: list, topics: list = TOPICS) -> str:
    context = f'A seguir, uma lista de resumos do hotel "{hotel}" por tópico e aspecto:'
    i = 0
    for topic in topics:
        for positive in [True, False]:
            title = "Aspectos positivos" if positive else "Aspectos negativos"
            title += f' do hotel no quesito "{topic}":\n'
            context += title + responses[i] + "\n"
            i += 1
    return context


def make_query_summary_full(hotel: str, topics: list) -> str:
    query = f'Faça um resumo do hotel "{hotel}" com base nos seguintes tópicos:\n'
    for i, t in enumerate(topics):
        query += f"{i+1}. {t}\n"
    query += "Para cada tópico, escreva um parágrafo com os principais aspectos positivos e negativos do hotel com base em suas avaliações."
    return query


def make_query_summary_topic(hotel: str, topic: str, positive=True) -> str:
    if positive:
        query = "Quais os aspectos positivos "
    else:
        query = "Quais os aspectos negativos "
    query += f'do hotel "{hotel}" no quesito "{topic}"?'
    return query


def make_prompt_summary_full(
    hotel_name: str,
    df: pd.DataFrame,
    n_docs: int = N_DOCS_MAX,
    topics: list = TOPICS,
):
    docs = make_docs(df[df.nome == hotel_name], n_docs=n_docs)
    context = make_context_summary(docs)
    query = make_query_summary_full(hotel_name, topics)
    prompt = PROMPT_SUMMARY.format(context=context, query=query)
    return prompt, context


def make_prompt_summary_full_v2(
    hotel_name: str,
    vector_store: FAISS,
    n_responses: int = N_RESPONSES_FULL,
    topics: list = TOPICS,
):
    filter_hotel = {"nome": {"$eq": hotel_name}}
    query = make_query_summary_full(hotel_name, topics)
    docs, _ = query_index(
        query=query,
        vector_store=vector_store,
        filter=filter_hotel,
        n_responses=n_responses,
    )
    context = make_context_summary(docs)
    prompt = PROMPT_SUMMARY.format(context=context, query=query)
    return prompt, context


def make_prompts_summary_topic(
    hotel_name: str,
    vector_store: FAISS,
    n_responses: int = N_RESPONSES_TOPIC,
    topics: list = TOPICS,
):
    filter_hotel = {"nome": {"$eq": hotel_name}}
    prompts = []
    contexts = []
    for i, topic in enumerate(topics):
        for positive in [True, False]:
            print(f"{hotel_name=} {i=} {topic=} {positive=}")
            if positive:
                filter_review = {"nota_avaliacao": {"$gte": 3}}
            else:
                filter_review = {"nota_avaliacao": {"$lte": 3}}
            filter_final = {**filter_hotel, **filter_review}
            query = make_query_summary_topic(
                hotel=hotel_name, topic=topic, positive=positive
            )
            docs, _ = query_index(
                query=query,
                vector_store=vector_store,
                filter=filter_final,
                n_responses=n_responses,
            )
            context = make_context_summary(docs)
            prompt = PROMPT_SUMMARY.format(context=context, query=query)
            contexts.append(context)
            prompts.append(prompt)
    return prompts, contexts


def make_prompt_final(
    hotel_name: str,
    responses: list,
    topics: list = TOPICS,
):
    context = make_context_final(hotel=hotel_name, responses=responses, topics=topics)
    query = make_query_summary_full(hotel_name, topics)
    prompt = PROMPT_SUMMARY.format(context=context, query=query)
    return prompt, context
