import psycopg
import os
from sqlalchemy import create_engine

PGSQL_USER = os.environ["PGSQL_USER"]
PGSQL_TOKEN = os.environ["PGSQL_TOKEN"]
PGSQL_DATABASE = os.environ["PGSQL_DATABASE"]


class PostgresConnection:
    def __init__(self):
        self.conn = None
        self.cursor = None

    def __enter__(self) -> psycopg.Connection:
        try:
            # Connect to the PostgreSQL server
            print("Attempting to connect...")
            self.conn = psycopg.connect(
                "dbname="
                + PGSQL_DATABASE
                + " user="
                + PGSQL_USER
                + " password="
                + PGSQL_TOKEN
            )
            # Create a cursor
            self.cursor = self.conn.cursor()
            return self.conn, self.cursor
        except (Exception, psycopg.DatabaseError) as error:
            print(error)
            if self.conn is not None:
                self.conn.close()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.cursor is not None:
                self.cursor.close()
            if self.conn is not None:
                self.conn.close()
                print("DB Conn closed.")
        except Exception as e:
            print(f"Error closing connection: {e}")


def bulk_insert(embeddings):
    # load data
    print(f"Loading {len(embeddings)} rows")
    with PostgresConnection() as (connection, cursor):
        with cursor.copy(
            "COPY items (embedding) FROM STDIN WITH (FORMAT BINARY)"
        ) as copy:
            copy.set_types(["vector"])
            for i, embedding in enumerate(embeddings):
                copy.write_row([embedding])

    print("\nSuccess!")


# Create an engine
DATABASE_URL = (
    f"postgresql+psycopg://{PGSQL_USER}:{PGSQL_TOKEN}@localhost/{PGSQL_DATABASE}"
)
engine = create_engine(DATABASE_URL)

if __name__ == "__main__":
    with PostgresConnection() as (connection, cursor):
        result = cursor.execute("select current_user;").fetchall()
        print(result)
        result = cursor.execute(
            "SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 1;"
        ).fetchall()
        print(result)
