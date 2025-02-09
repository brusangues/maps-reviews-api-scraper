# pip install python-telegram-bot

import os
import traceback
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackContext,
    MessageHandler,
    filters,
    ContextTypes,
)
import torch
import json

from llms.models import load_model, load_embedding, query_model_async
from llms.rag import load_index, query_index, PROMPT


LLM = None
INDEX = None
FILTER = {}


# Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LLM, INDEX, FILTER
    del LLM, INDEX, FILTER
    torch.cuda.empty_cache()
    LLM = None
    INDEX = None
    FILTER = {}
    print("\nstart_command")
    await update.message.reply_text("Oi, eu sou um bot!")


async def echo_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("\necho_command")
    await update.message.reply_text(update.message.text)


async def load_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LLM
    del LLM
    torch.cuda.empty_cache()
    print("\nload_command")
    args = update.message.text.replace("/load", "").strip().split()
    print(f"{args=}")
    LLM, model_name = load_model(*args)
    await update.message.reply_text(f"Modelo carregado: {model_name}")


async def rag_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global INDEX
    del INDEX
    torch.cuda.empty_cache()
    print("\nrag_command")
    model_alias = update.message.text.replace("/rag", "").strip()
    print(f"{model_alias=}")
    embedding, model_name = load_embedding(model_alias)
    INDEX = load_index(embedding)
    await update.message.reply_text(f"Modelo carregado: {model_name}")


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
    elif LLM is not None and INDEX is not None:
        results, context = query_index(INDEX, text, FILTER)
        await update.message.reply_text(context)
        prompt = PROMPT.format(
            context=context,
            question=text,
        )
        return await query_model_async(LLM, prompt)
    elif LLM is not None:
        return await query_model_async(LLM, text)
    return (
        "Nenhum modelo está carregado. "
        "Use o comando /load para carregar um modelo de linguagem. "
        "Em seguida, use o comando /rag para carregar o índice de avaliações de hotéis. "
        "Se desejar, use o comando /filter para definir um filtro para as avaliações."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("\nhandle_message")
    print(f"{update=}\n{context=}")

    try:
        message_type = update.message.chat.type
        text = update.message.text
        print(f"User ({update.message.chat.id}) in {message_type}: {text}")

        response = await handle_response(update, text)

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
    app.add_handler(CommandHandler("echo", echo_command))
    app.add_handler(CommandHandler("load", load_command))
    app.add_handler(CommandHandler("rag", rag_command))
    app.add_handler(CommandHandler("filter", filter_command))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    # Errors
    app.add_error_handler(error)

    print("Polling...")
    app.run_polling(poll_interval=3)
