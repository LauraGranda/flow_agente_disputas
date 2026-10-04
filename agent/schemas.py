"""Contratos de datos (Pydantic) del agente de disputas.

- DecisionAgente: el único formato que el LLM puede devolver. Es estricto
  (extra="forbid"): si el JSON trae campos de más o valores fuera de rango,
  no valida y se descarta.
- PaqueteEvidencia: la evidencia de solo lectura que arma agent.tools.
- Chequeo / AccionFinal / ResultadoPolitica: la salida del motor de políticas.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Causa = Literal["aclarable", "error_vinculacion_placa", "monto", "fraude_probable", "otra"]

ESTADO_PENDIENTE_APROBACION = "pendiente_aprobacion_humana"
ESTADO_BORRADOR = "borrador"


class _Estricto(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- Salida del LLM ---------------------------------------------------------

class Accion(_Estricto):
    # El tipo es texto libre a propósito: el filtro de acciones permitidas vive
    # en agent.policy, para que una acción inventada quede registrada como
    # descartada en vez de tumbar toda la decisión.
    tipo: str = Field(min_length=1, max_length=64)
    monto_cop: Optional[int] = Field(default=None, ge=0)


class DecisionAgente(_Estricto):
    causa: Causa
    evidencia_citada: list[str] = Field(default_factory=list)
    confianza: float = Field(ge=0, le=1)
    acciones: list[Accion] = Field(default_factory=list)
    borrador_respuesta_cliente: str = Field(max_length=2000)
    razonamiento_breve: str = Field(max_length=1000)
    alertas: list[str] = Field(default_factory=list)


class PropuestaAgente(DecisionAgente):
    """Input de la herramienta proponer_decision: la decisión + lo que el LLM le diría al cliente.

    respuesta_cliente es solo una propuesta: el orquestador decide si el cliente
    la ve (aclaraciones permitidas) o si se reemplaza por un mensaje fijo.
    """

    respuesta_cliente: str = Field(min_length=1, max_length=2000)
    # Debe apoyarse en evidencia: citar al menos una consulta (si no, se le pide corregir).
    evidencia_citada: list[str] = Field(min_length=1)


# --- Evidencia ---------------------------------------------------------------

class PaqueteEvidencia(_Estricto):
    """Evidencia de una transacción. Cada bloque trae su propio 'evidencia_id'."""

    tx_id: str
    transaccion: dict
    foto_placa: dict
    venta_fcc: dict
    estado_push: dict
    senales_fraude: dict

    def bloques(self) -> list[dict]:
        return [self.transaccion, self.foto_placa, self.venta_fcc, self.estado_push, self.senales_fraude]

    def ids(self) -> set[str]:
        return {b["evidencia_id"] for b in self.bloques()}


# --- Salida del motor de políticas ------------------------------------------

class Chequeo(_Estricto):
    nombre: str
    aprobado: bool
    motivo: str


class AccionFinal(_Estricto):
    tipo: Literal["enviar_aclaracion", "recomendar_reembolso", "escalar_analista", "abrir_caso_mlops"]
    monto_cop: Optional[int] = None
    estado: Literal["pendiente_aprobacion_humana", "borrador"]


class ResultadoPolitica(_Estricto):
    chequeos: list[Chequeo]
    acciones_finales: list[AccionFinal]
    escalado: bool
    motivos_escalamiento: list[str]
    causa_final: Optional[Causa] = None  # causa tras las correcciones de coherencia con la evidencia
