"""Entry point for running FloodLens locally."""
from app import create_app
from config import Config

app = create_app()

if __name__ == "__main__":
    Config.ensure_dirs()
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
