import pandas as pd
from langchain_core.documents import Document
from llms.rag import load_data
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


def prep_data(df: pd.DataFrame) -> tuple[list, list]:
    hotel_review_counts = (
        df.groupby("nome")
        .agg(
            count_=("text", "count"),
        )
        .sort_values("count_", ascending=False)
        .reset_index()
    )

    hotels_small = hotel_review_counts.query("count_>=10 & count_<=1000")
    hotels_big = hotel_review_counts.query("count_>1000")

    return hotels_small, hotels_big


def make_docs(df_hotel) -> list[tuple[Document, None]]:
    print("make_docs...")
    docs = []
    for i, (id, row) in enumerate(df_hotel.iterrows()):
        metadata = row.to_dict()
        text = metadata.pop("text")
        doc = Document(page_content=text, metadata=metadata)
        docs.append((doc, None))
    print(f"{len(docs)=}")
    return docs


def make_context_summary(docs) -> str:
    meta = docs[0][0].__dict__["metadata"]
    context = (
        f"Hotel: {meta['nome']}, {int(meta['estrelas'])} Estrelas.\n"
        f"Região:{meta['regiao']}; Estado:{meta['estado']}; Cidade:{meta['cidade']}\n"
        f"Tipo:{meta['subcategoria']}; Classificação:{meta['classificacao_geral']}; Quantidade Avaliações:{int(meta['quantidade_avaliacoes'])}\n"
        "\n"
    )

    for i, (doc, score) in enumerate(docs):
        context_i = format_context(i, doc, score, include_hotel_context=False)
        context += context_i

    return context


def make_query_summary_full(hotel: str, topics: list) -> str:
    query = f'Faça um resumo do hotel "{hotel}" com base nos seguintes tópicos:\n'
    for i, t in enumerate(topics):
        query += f"{i+1}. {t}\n"
    query += "Para cada tópico, escreva um parágrafo com os principais aspectos positivos e negativos do hotel com base em suas avaliações."
    return query


def make_query_summary_topic(hotel: str, topic: str, positive=True) -> str:
    if positive:
        query = "Quais os aspectos positivos (avaliações com nota 3 ou mais) "
    else:
        query = "Quais os aspectos negativos (avaliações com nota 3 ou menos) "
    query += f'do hotel "{hotel}" no quesito "{topic}"?'
    return query


def make_prompt_summary_full(hotel_name: str, df: pd.DataFrame):
    docs = make_docs(df[df.nome == hotel_name])
    context = make_context_summary(docs)
    query = make_query_summary_full(hotel_name, TOPICS)
    prompt = PROMPT_SUMMARY.format(context=context, query=query)
    print(prompt)
    return prompt, context


def make_summaries():
    df = load_data()
    hotels_small, hotels_big = prep_data(df)
    for hotel_name in hotels_small:
        prompt = make_prompt_summary_full(hotel_name, df)
