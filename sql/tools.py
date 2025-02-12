import os
from sqlalchemy import create_engine
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit


PGSQL_USER = os.environ["PGSQL_USER"]
PGSQL_TOKEN = os.environ["PGSQL_TOKEN"]
PGSQL_DATABASE = os.environ["PGSQL_DATABASE"]

DATABASE_URL = (
    f"postgresql+psycopg://{PGSQL_USER}:{PGSQL_TOKEN}@localhost/{PGSQL_DATABASE}"
)
engine = create_engine(DATABASE_URL)

db = SQLDatabase(engine)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=100,
    timeout=10,
    max_retries=2,
    # other params...
)


toolkit = SQLDatabaseToolkit(db=db, llm=llm)
toolkit.get_tools()
