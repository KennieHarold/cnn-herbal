from server.app import app
from dotenv import load_dotenv
import os

BASEDIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASEDIR, '.env'))

if __name__ == "__main__":
    if os.getenv("DEBUG") == "True":
        app.run(debug=True, port=8080)
    else:
        app.run()