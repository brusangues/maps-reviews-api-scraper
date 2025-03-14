import time
import re
import pandas as pd

from llms.models import query_model

### **Modelo de Prompt de Avaliação baseado no G-Eval**

EVALUATION_PROMPT_TEMPLATE = """
Você receberá um RESUMO de um hotel feito a partir de avaliações de usuários,
contendo os seguintes tópicos: Infraestrutura e Acomodações; Atendimento e Serviço; Localização e Acessibilidade; Alimentação e Bebidas; Experiência e Entretenimento; Custo-benefício e Políticas.
Sua tarefa é avaliar o RESUMO com base em um critério específico, comparando-o com o TEXTO ORIGINAL.
Certifique-se de ler e entender essas instruções com muito cuidado.
Mantenha este documento aberto enquanto revisa e consulte-o conforme necessário.
Seja bastante rigoroso na avaliação.

**CRITÉRIO DE AVALIAÇÃO:**

{criteria}
Seja bastante rigoroso na avaliação.

**ETAPAS DE AVALIAÇÃO:**

{steps}

**TEXTO ORIGINAL:**

{document}

**RESUMO:**

{summary}

**FORMULÁRIO DE AVALIAÇÃO:** (responda APENAS com o número da pontuação do critério de avaliação, sem explicar)

- Nota de {metric_name}: """

### **Métrica 1: Relevância**

RELEVANCY_SCORE_CRITERIA = """
Relevância (1-5) - seleção do conteúdo importante do TEXTO ORIGINAL.
O RESUMO deve incluir apenas informações essenciais do TEXTO ORIGINAL.
O avaliador deve penalizar um RESUMO que contenha redundâncias ou informações excessivas.

"""

RELEVANCY_SCORE_STEPS = """ 
1. Leia atentamente o RESUMO e o TEXTO ORIGINAL.
2. Compare o RESUMO com o TEXTO ORIGINAL e identifique os principais pontos de informação.
3. Avalie o quanto o RESUMO cobre os principais pontos contidos no TEXTO ORIGINAL e se contém informações irrelevantes ou redundantes.
4. Atribua uma pontuação de relevância em uma escala de 1 a 5.
"""
### **Métrica 2: Coerência**

COHERENCE_SCORE_CRITERIA = """
Coerência (1-5) - qualidade coletiva de todas as frases.
O RESUMO deve ser bem estruturado e organizado.
O RESUMO não deve ser apenas um amontoado de informações relacionadas, mas deve
construir uma sequência lógica e coerente de informações sobre um determinado tópico.
"""

COHERENCE_SCORE_STEPS = """
1. Leia o TEXTO ORIGINAL com atenção e identifique o tópico principal e os pontos-chave.
2. Leia o RESUMO e compare-o com o TEXTO ORIGINAL. Verifique se o RESUMO cobre o tópico principal e os pontos-chave,
e se os apresenta de forma clara e lógica.
3. Atribua uma pontuação para coerência em uma escala de 1 a 5, onde 1 é a menor e 5 a maior, com base nos critérios de avaliação.
"""

### **Métrica 3: Consistência**

CONSISTENCY_SCORE_CRITERIA = """
Consistência (1-5) - alinhamento factual entre o RESUMO e o documento original.
Um RESUMO factualmente consistente contém apenas afirmações confirmadas pelo TEXTO ORIGINAL.
O avaliador deve penalizar um RESUMO que contenha informações falsas ou inventadas.
"""

CONSISTENCY_SCORE_STEPS = """
1. Leia o TEXTO ORIGINAL cuidadosamente e identifique os principais fatos e detalhes apresentados.
2. Leia o RESUMO e compare-o com o TEXTO ORIGINAL. Verifique se o RESUMO contém erros factuais não suportados pelo TEXTO ORIGINAL.
3. Atribua uma pontuação para consistência em uma escala de 1 a 5, com base nos critérios de avaliação.
"""

### **Métrica 4: Fluência**

FLUENCY_SCORE_CRITERIA = """
Fluência (1-3) - qualidade do RESUMO em termos de gramática, ortografia, pontuação, escolha de palavras e estrutura das frases.
Penalize muito as REPETIÇÕES.
1: Ruim. O RESUMO contém muitos erros ou repetições que dificultam a compreensão ou o fazem soar pouco natural.
2: Regular. O RESUMO tem alguns erros ou repetições que afetam a clareza ou fluidez do texto, mas os principais pontos ainda são compreensíveis.
3: Bom. O RESUMO tem poucos ou nenhum erro e é fácil de ler e acompanhar.
"""

FLUENCY_SCORE_STEPS = """
Leia o RESUMO e avalie sua fluência com base nos critérios fornecidos.
Atribua uma pontuação de fluência de 1 a 3, com base nas diretivas dos critérios de avaliação.
"""

evaluation_metrics = {
    "Relevância": (RELEVANCY_SCORE_CRITERIA, RELEVANCY_SCORE_STEPS),
    "Coerência": (COHERENCE_SCORE_CRITERIA, COHERENCE_SCORE_STEPS),
    "Consistência": (CONSISTENCY_SCORE_CRITERIA, CONSISTENCY_SCORE_STEPS),
    "Fluência": (FLUENCY_SCORE_CRITERIA, FLUENCY_SCORE_STEPS),
}


def g_eval_scores(summaries: list, document: str, llm, sleep=1):
    print("g_eval_scores...")
    results = []
    scores = []

    for i, summary in enumerate(summaries):
        print(f"Evaluating summary {i}")
        for metric_name, (criteria, steps) in evaluation_metrics.items():
            print(f"{metric_name=}")
            prompt = EVALUATION_PROMPT_TEMPLATE.format(
                criteria=criteria,
                steps=steps,
                metric_name=metric_name,
                document=document,
                summary=summary,
            )
            response, info = query_model(llm, prompt)
            try:
                # score = int(response.strip())
                score = int(re.search(r"(\d+)", response).group(1))
            except Exception as e:
                print(e)
                score = None
            data = {
                "index": i,
                "summary": summary,
                "metric_name": metric_name,
                "score": score,
                "prompt": prompt,
                "response": response,
                "info": info,
            }
            results.append(data)
            scores.append(score)
            print(f"Sleeping {sleep} seconds...")
            time.sleep(sleep)
    score_final = pd.Series(scores).mean()
    print(f"{score_final=} {scores=}")
    return results, score_final
