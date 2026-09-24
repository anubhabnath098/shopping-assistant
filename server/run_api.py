"""Entrypoint for running the API server: `python run_api.py`."""
import uvicorn
from config.settings import get_settings


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("api.app:app", host=settings.api_host, port=settings.api_port, reload=False)