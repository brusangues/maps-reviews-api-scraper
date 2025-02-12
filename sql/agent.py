from langchain_google_genai import ChatGoogleGenerativeAI

# gemini-2.0-flash
# gemini-2.0-flash-lite-preview-02-05
# gemini-1.5-flash
# gemini-1.5-flash-8b
# gemini-1.5-pro


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=100,
    timeout=10,
    max_retries=2,
    # other params...
)

messages = [
    (
        "system",
        "Você é um assistente pessoal.",
    ),
    ("human", "Resumidamente, qual o sentido da vida?"),
]
print(llm.invoke(messages))
