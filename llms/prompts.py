PROMPT = """
Você é um assistente para tarefas de resposta a perguntas em português.
Leia a PERGUNTA e use os seguintes trechos de CONTEXTO recuperado de avaliações de hotéis para respondê-la.
Quanto menor a Similaridade da avaliação, mais ela se assemelha com a PERGUNTA.
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
"regiao": Região geográfica do Brasil onde o hotel está localizado. Valores possíveis: ["SUDESTE", "NORTE", "NORDESTE", "SUL", "CENTRO-OESTE"].
"endereco": Endereço completo do hotel, incluindo rua, número, bairro, cidade e estado.
"classificacao_geral": Nota geral do hotel baseada nas avaliações dos usuários, variando de 0.0 a 5.0.
"quantidade_avaliacoes": Número total de avaliações recebidas pelo hotel. Pode ser utilizado para filtrar hotéis grandes (>1000 avaliações).
"nota_avaliacao": Nota específica da avaliação, variando de 1 a 5.
"curtidas_avaliacao": Número de curtidas recebidas por uma avaliação.
"usuario_guia_local": Indica se o usuário que fez a avaliação é um Guia Local do Google Maps (1 para sim, 0 para não).
"data_avaliacao": Data em que a avaliação foi feita, no formato "YYYY-MM-DD".
"nota_quartos": Nota dada pelos usuários para os quartos do hotel, variando de 1 a 5.
"nota_localizacao": Nota dada pelos usuários para a localização do hotel, variando de 1 a 5.
"nota_servico": Nota dada pelos usuários para o serviço do hotel, variando de 1 a 5.
"avaliacao_recente": Indica se a avaliação foi feita dentro de 6 meses (1 para sim, 0 para não).
"numero_palavras_avaliacao": Quantidade de palavras contidas na avaliação do usuário.

Após o JSON, inclua a pergunta reformulada, excluindo a parte que foi usada para montar a query.
Siga os seguintes exemplos:
PERGUNTA: Qual o melhor hotel com 3 estrelas ou mais?
RESPOSTA: ```json {"estrelas": {"$gte": 3}} ``` Qual o melhor hotel?\n
PERGUNTA: Qual o melhor hotel no estado de SP?
RESPOSTA: ```json {"sigla_estado": {"$eq": "SP"}} ``` Qual o melhor hotel?\n
PERGUNTA: Quero um hotel de 4 estrelas pé na areia.
RESPOSTA: ```json {"estrelas": {"$eq": 4}} ``` Quero um hotel pé na areia.\n
PERGUNTA: Qual hotel tem o melhor café da manhã na cidade de Amparo, no estado de SP?
RESPOSTA: ```json {"cidade": {"$eq": "Amparo"}, "sigla_estado": {"$eq": "SP"}} ``` Qual hotel tem o melhor café da manhã?'\n
FIM DOS EXEMPLOS!!!\n
Não explique a resposta e responda apenas com o JSON seguido da pergunta reformulada.\n
"""


def format_context(i, res, score=None, include_hotel_context=True):
    meta = res.metadata
    text = res.page_content
    review_i = i + 1

    score_context = f", Similaridade: {score:0.3f}" if score is not None else ""

    if include_hotel_context:
        hotel_context = (
            f"Hotel: {meta['nome']}, {int(meta['estrelas'])} Estrelas.\n"
            f"Região:{meta['regiao']}; Estado:{meta['estado']}; Cidade:{meta['cidade']}\n"
            f"Tipo:{meta['subcategoria']}; Classificação:{meta['classificacao_geral']}; Quantidade Avaliações:{int(meta['quantidade_avaliacoes'])}\n"
        )
    else:
        hotel_context = ""

    outras_notas = ""
    nomes_outras_notas = {
        "nota_quartos": "Quartos",
        "nota_localizacao": "Localização",
        "nota_servico": "Serviço",
    }
    for key, name in nomes_outras_notas.items():
        if meta[key] == meta[key]:  # Checando se é NaN
            outras_notas += f"; Nota {name}:{int(meta[key])}"

    local_guide_context = (
        "; Usuário é guia local"
        if meta["usuario_guia_local"]
        else "; Usuário não é guia local"
    )

    context = (
        f" - Avaliação {review_i}{score_context}\n"
        f"{hotel_context}"
        f"Nota:{int(meta['nota_avaliacao'])}; Curtidas:{int(meta['curtidas_avaliacao'])}"
        f"{local_guide_context}{outras_notas}\n"
        f"Avaliação: {text}\n\n"
    )
    return context
