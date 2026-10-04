from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from agent import llm, orchestrator

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def entorno_aislado(monkeypatch, tmp_path):
    """Fuerza el modo simulado y escribe la cola en un archivo temporal."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MODO_LLM", raising=False)
    monkeypatch.setattr(orchestrator, "COLA_PATH", tmp_path / "cola_analista.json")
    assert llm.modo_actual()["modo"] == "simulado"


def abrir_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception, at.exception
    return at
