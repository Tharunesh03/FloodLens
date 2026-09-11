"""WSGI entrypoint for production servers (gunicorn, uwsgi, etc.)."""
from app import create_app

application = create_app()
app = application
