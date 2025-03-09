import time
import re
import pandas as pd
from tqdm import tqdm

from llms.models import query_model

### **Modelo de Prompt de Avaliação baseado no G-Eval**

EVALUATION_PROMPT_TEMPLATE = """
Você receberá um resumo escrito para um artigo. Sua tarefa é avaliar o resumo com base em um critério específico.  
Certifique-se de ler e entender essas instruções com muito cuidado.  
Mantenha este documento aberto enquanto revisa e consulte-o conforme necessário.  

**Critério de Avaliação:**  

{criteria}  

**Etapas de Avaliação:**  

{steps}  

**Exemplo:**  

**Texto Original:**  

{document}  

**Resumo:**  

{summary}  

**Formulário de Avaliação (responda APENAS com o número da pontuação, sem explicar):**  

- {metric_name}:
"""

### **Métrica 1: Relevância**

RELEVANCY_SCORE_CRITERIA = """
Relevância (1-5) - seleção do conteúdo importante do texto original.  
O resumo deve incluir apenas informações essenciais do documento-fonte.  
Os avaliadores devem penalizar resumos que contenham redundâncias ou informações excessivas.  
"""

RELEVANCY_SCORE_STEPS = """ 
1. Leia atentamente o resumo e o documento-fonte.  
2. Compare o resumo com o documento-fonte e identifique os principais pontos do artigo.  
3. Avalie o quanto o resumo cobre os principais pontos do artigo e se contém informações irrelevantes ou redundantes.  
4. Atribua uma pontuação de relevância de 1 a 5.  
"""
### **Métrica 2: Coerência**

COHERENCE_SCORE_CRITERIA = """
Coerência (1-5) - qualidade coletiva de todas as frases.  
Esta dimensão segue a diretriz de qualidade do DUC sobre estrutura e coerência,  
segundo a qual "o resumo deve ser bem estruturado e organizado.  
O resumo não deve ser apenas um amontoado de informações relacionadas, mas deve  
construir uma sequência lógica e coerente de informações sobre um tópico."  
"""

COHERENCE_SCORE_STEPS = """
1. Leia o artigo com atenção e identifique o tópico principal e os pontos-chave.  
2. Leia o resumo e compare-o com o artigo. Verifique se o resumo cobre o tópico principal e os pontos-chave,  
e se os apresenta de forma clara e lógica.  
3. Atribua uma pontuação para coerência em uma escala de 1 a 5, onde 1 é a menor e 5 a maior, com base nos critérios de avaliação.  
"""

### **Métrica 3: Consistência**

CONSISTENCY_SCORE_CRITERIA = """
Consistência (1-5) - alinhamento factual entre o resumo e o documento original.  
Um resumo factualmente consistente contém apenas afirmações confirmadas pelo documento-fonte.  
Os avaliadores devem penalizar resumos que contenham informações falsas ou inventadas.  
"""

CONSISTENCY_SCORE_STEPS = """
1. Leia o artigo cuidadosamente e identifique os principais fatos e detalhes apresentados.  
2. Leia o resumo e compare-o com o artigo. Verifique se o resumo contém erros factuais não suportados pelo documento-fonte.  
3. Atribua uma pontuação para consistência com base nos critérios de avaliação.  
"""

### **Métrica 4: Fluência**

FLUENCY_SCORE_CRITERIA = """
Fluência (1-3) - qualidade do resumo em termos de gramática, ortografia, pontuação, escolha de palavras e estrutura das frases.  
1: Ruim. O resumo contém muitos erros que dificultam a compreensão ou o fazem soar pouco natural.  
2: Regular. O resumo tem alguns erros que afetam a clareza ou fluidez do texto, mas os principais pontos ainda são compreensíveis.  
3: Bom. O resumo tem poucos ou nenhum erro e é fácil de ler e acompanhar.  
"""

FLUENCY_SCORE_STEPS = """
Leia o resumo e avalie sua fluência com base nos critérios fornecidos. Atribua uma pontuação de fluência de 1 a 3.  
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
