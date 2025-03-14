"""
start - inicializar o ambiente
reset - reiniciar o ambiente
echo - retorna a mensagem do usuário
help - ajuda
load - carrega um modelo de linguagem
rag - carrega modelo de embedding e índice de busca semântica
filter - configura um filtro para o índice de busca semântica
"""

# pip install python-telegram-bot
import os
import traceback
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
import torch
import json

from llms.models import (
    load_model,
    query_model,
    models_text,
    models_embedding,
)
from llms.rag import load_rag, query_index, query_make_filter, PROMPT


LLM = None
INDEX = None
FILTER = {}
MESSAGE_LIMIT = 4095


# Commands
def initial_start():
    global LLM, INDEX, FILTER
    print("Initial start")
    FILTER = {}
    LLM, model_name, max_new_tokens = load_model("gemini-2.0-flash-lite")
    print(f"Modelo carregado: {model_name}\nCom máximo de tokens: {max_new_tokens}")
    INDEX, rag_alias, embeddings_name = load_rag("google-ip")
    print(f"Índice carregado: {rag_alias}\nModelo de embedding: {embeddings_name}")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("\nstart_command")
    global LLM, INDEX, FILTER
    del LLM, INDEX, FILTER
    torch.cuda.empty_cache()
    FILTER = {}
    LLM, model_name, max_new_tokens = load_model()
    await update.message.reply_text(
        f"Modelo carregado: {model_name}\nCom máximo de tokens: {max_new_tokens}"
    )
    INDEX, rag_alias, embeddings_name = load_rag("google-ip")
    await update.message.reply_text(
        f"Índice carregado: {rag_alias}\nModelo de embedding: {embeddings_name}"
    )
    await update.message.reply_text("Oi, eu sou um bot. Ambiente inicializado!")


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LLM, INDEX, FILTER
    print("\nreset_command")
    del LLM, INDEX, FILTER
    torch.cuda.empty_cache()
    LLM = None
    INDEX = None
    FILTER = {}
    await update.message.reply_text("Ambiente reiniciado!")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("\nhelp_command")
    await update.message.reply_text(
        f"Modelos de linguagem disponíveis:\n{models_text}\n"
        f"Modelos de embedding disponíveis:\n{models_embedding}\n"
        f"Comandos disponíveis: start, echo, load, rag ,filter, help"
    )


async def echo_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("\necho_command")
    await update.message.reply_text(update.message.text)


async def load_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LLM
    print("\nload_command")
    del LLM
    args = update.message.text.replace("/load", "").strip().split()
    print(f"{args=}")
    LLM, model_name, max_new_tokens = load_model(*args)
    await update.message.reply_text(
        f"Modelo carregado: {model_name}\nCom máximo de tokens: {max_new_tokens}"
    )


async def rag_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global INDEX
    print("\nrag_command")
    del INDEX
    rag_alias = update.message.text.replace("/rag", "").strip()
    INDEX, rag_alias, embeddings_name = load_rag(rag_alias)
    await update.message.reply_text(
        f"Índice carregado: {rag_alias}\nModelo de embedding: {embeddings_name}"
    )


async def filter_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global FILTER
    print("\nfilter_command")
    filter = update.message.text.replace("/filter", "").strip()
    print(f"{filter=}")
    reply_text = ""
    if len(filter) < 2:
        FILTER = {}
        reply_text = "Filtro inválido. "
    else:
        try:
            FILTER = json.loads(filter)
        except Exception as e:
            print(e)
            FILTER = {}
            reply_text = "Filtro inválido. "
    reply_text += f"Filtro: {FILTER}"
    await update.message.reply_text(reply_text)


# Messages
async def handle_response(update: Update, text: str) -> str:
    global LLM, FILTER, INDEX
    if "oi" == text.lower():
        return "Oi de novo, eu sou um bot!"
    elif LLM is not None and INDEX is not None and len(FILTER.keys()) == 0:
        filter, query_updated = query_make_filter(LLM, text)
        await update.message.reply_text(
            f"Filtro criado dinamicamente: {filter}\nQuery atualizada: {query_updated}"
        )
        results, context = query_index(INDEX, query_updated, filter)
        response = f"Resultado da busca no índice:\n{context}"
        print(f"Response length: {len(response)}/{MESSAGE_LIMIT}")
        response = response[:MESSAGE_LIMIT]
        await update.message.reply_text(response)
        prompt = PROMPT.format(
            context=context,
            question=text,
        )
        return "Resposta final:\n" + query_model(LLM, prompt)[0]
    elif LLM is not None and INDEX is not None:
        results, context = query_index(INDEX, text, FILTER)
        response = f"Resultado da busca no índice:\n{context}"
        print(f"Response length: {len(response)}/{MESSAGE_LIMIT}")
        response = response[:MESSAGE_LIMIT]
        await update.message.reply_text(response)
        prompt = PROMPT.format(
            context=context,
            question=text,
        )
        return "Resposta final:\n" + query_model(LLM, prompt)[0]
    elif LLM is not None:
        return query_model(LLM, text)[0]
    return (
        "Nenhum modelo está carregado. "
        "Use o comando /load para carregar um modelo de linguagem. "
        "Em seguida, use o comando /rag para carregar o índice de avaliações de hotéis. "
        "Se desejar, use o comando /filter para definir um filtro para as avaliações manualmente."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("\nhandle_message")
    print(f"{update=}\n{context=}")

    try:
        message_type = update.message.chat.type
        text = update.message.text
        print(f"User ({update.message.chat.id}) in {message_type}: {text}")

        response = await handle_response(update, text)
        print(f"Response length: {len(response)}/{MESSAGE_LIMIT}")
        response = response[:MESSAGE_LIMIT]

        print(f"Bot: {response}")
        await update.message.reply_text(response)

    except Exception as e:
        print(e)
        traceback.print_exc()
        raise e


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("\nerror")
    print(f"{update=}\n{context.error=}")
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Desculpe, ocorreu um erro!",
    )


# Main
if __name__ == "__main__":
    print("Starting bot...")
    app = Application.builder().token(os.environ["TELEGRAM_TOKEN"]).build()

    # Commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("echo", echo_command))
    app.add_handler(CommandHandler("load", load_command))
    app.add_handler(CommandHandler("rag", rag_command))
    app.add_handler(CommandHandler("filter", filter_command))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    # Errors
    app.add_error_handler(error)

    initial_start()

    print("Polling...")
    app.run_polling(poll_interval=3)
