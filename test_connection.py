import os
from dotenv import load_dotenv
import psycopg2

# Load environment variables from .env file
load_dotenv()
DB_URL = os.getenv("DB_URL")

try:
    # Establish a connection
    conn = psycopg2.connect(DB_URL)
    print("Connection successful!")

    # Close the connection
    conn.close()
except Exception as e:
    print(f"Error: {e}")
