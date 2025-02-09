import psycopg
import os

PGSQL_USER = os.environ["PGSQL_USER"]
PGSQL_TOKEN = os.environ["PGSQL_TOKEN"]
PGSQL_DATABASE = os.environ["PGSQL_DATABASE"]


def execute_postgres(query):
    conn = None
    try:

        # connect to the PostgreSQL server
        print("Attempting to connect...")
        conn = psycopg.connect(
            "dbname="
            + PGSQL_DATABASE
            + " user="
            + PGSQL_USER
            + " password="
            + PGSQL_TOKEN
        )

        # create a cursor
        cursor = conn.cursor()

        # execute a statement
        cursor.execute(query)
        result = cursor.fetchone()
        print(result)

        # close the communication with the PostgreSQL
        cursor.close()
    except (Exception, psycopg.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print("DB Conn closed.")


if __name__ == "__main__":
    execute_postgres("select current_user;")
    execute_postgres("SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 1;")
