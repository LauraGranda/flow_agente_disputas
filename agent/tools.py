"""Herramientas de consulta de evidencia (SOLO LECTURA) sobre data/ledger.json.

Cada función recibe un tx_id y devuelve un dict nuevo con un 'evidencia_id'
estable (p. ej. "fcc:TX-1002") que el LLM debe citar en evidencia_citada.
Ninguna función escribe en disco ni en ningún sistema; se devuelven copias
para que nadie pueda alterar el ledger en memoria.
"""

import copy
import json
from functools import lru_cache
from pathlib import Path

from agent.schemas import PaqueteEvidencia

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LEDGER_PATH = DATA_DIR / "ledger.json"
PLATES_DIR = DATA_DIR / "plates"
UMBRAL_OCR = 0.85


class TransaccionNoEncontrada(KeyError):
    pass


@lru_cache(maxsize=1)
def _ledger() -> dict[str, dict]:
    data = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    return {tx["tx_id"]: tx for tx in data["transacciones"]}


def _tx(tx_id: str) -> dict:
    try:
        return copy.deepcopy(_ledger()[tx_id])
    except KeyError:
        raise TransaccionNoEncontrada(tx_id) from None


def listar_transacciones() -> list[dict]:
    """Resumen de las transacciones disponibles (para elegir sobre cuál reclamar)."""
    return [
        {"tx_id": tx["tx_id"], "placa_cuenta": tx["placa_cuenta"], "estacion": tx["estacion"],
         "fecha_hora": tx["fecha_hora"], "monto_capturado_cop": tx["monto_capturado_cop"]}
        for tx in copy.deepcopy(list(_ledger().values()))
    ]


def get_transaccion(tx_id: str) -> dict:
    tx = _tx(tx_id)
    return {
        "evidencia_id": f"tx:{tx_id}",
        "tx_id": tx_id,
        "cuenta": tx["cuenta"],
        "placa_cuenta": tx["placa_cuenta"],
        "estacion": tx["estacion"],
        "surtidor": tx["surtidor"],
        "fecha_hora": tx["fecha_hora"],
        "hold_cop": tx["hold_cop"],
        "monto_capturado_cop": tx["monto_capturado_cop"],
        "monto_liberado_cop": tx["hold_cop"] - tx["monto_capturado_cop"],
    }


def get_foto_placa(tx_id: str) -> dict:
    tx = _tx(tx_id)
    return {
        "evidencia_id": f"foto:{tx_id}",
        "ruta_imagen": str(PLATES_DIR / f"{tx_id}.png"),
        "placa_leida": tx["placa_foto_anpr"],
        "placa_cuenta": tx["placa_cuenta"],
        "coincide_con_cuenta": tx["placa_foto_anpr"] == tx["placa_cuenta"],
        "confianza_ocr": tx["confianza_ocr"],
        "ocr_bajo_umbral": tx["confianza_ocr"] < UMBRAL_OCR,
    }


def get_venta_fcc(tx_id: str) -> dict:
    """Venta registrada por el controlador de surtidores (FCC) de la estación."""
    tx = _tx(tx_id)
    calculado = round(tx["galones"] * tx["precio_galon_cop"])
    return {
        "evidencia_id": f"fcc:{tx_id}",
        "estacion": tx["estacion"],
        "surtidor": tx["surtidor"],
        "galones": tx["galones"],
        "precio_galon_cop": tx["precio_galon_cop"],
        "monto_venta_cop": calculado,
        "cuadra_con_captura": calculado == tx["monto_capturado_cop"],
    }


def get_estado_push(tx_id: str) -> dict:
    tx = _tx(tx_id)
    return {
        "evidencia_id": f"push:{tx_id}",
        "push_entregado": tx["push_entregado"],
    }


def get_senales_fraude(tx_id: str) -> dict:
    tx = _tx(tx_id)
    return {
        "evidencia_id": f"fraude:{tx_id}",
        "dispositivo_nuevo": tx["dispositivo_nuevo"],
        "senales": tx["senales_fraude"],
    }


def get_evidencia(tx_id: str) -> PaqueteEvidencia:
    """Reúne toda la evidencia de una transacción en un solo paquete."""
    return PaqueteEvidencia(
        tx_id=tx_id,
        transaccion=get_transaccion(tx_id),
        foto_placa=get_foto_placa(tx_id),
        venta_fcc=get_venta_fcc(tx_id),
        estado_push=get_estado_push(tx_id),
        senales_fraude=get_senales_fraude(tx_id),
    )
