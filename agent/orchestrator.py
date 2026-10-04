"""Orquestador determinístico del Asistente Flow.

procesar_turno(tx_id, mensaje, historial) — un turno del chat del cliente:
1. Lee la evidencia completa con agent.tools (la usa el motor de políticas).
2. Abre una sesión con el LLM (agent.llm). El LLM PIDE herramientas de lectura;
   este módulo las ejecuta (solo sobre la transacción elegida) y le devuelve el
   resultado. Máximo MAX_LLAMADAS consultas de lectura por turno.
3. El LLM termina llamando a proponer_decision; su input se valida con Pydantic.
4. La decisión pasa por agent.policy.
5. Este módulo decide qué ve el cliente y el resultado del turno:
     verde  "Respondido al cliente"  -> aclaración permitida (se muestra la respuesta del LLM)
     ámbar  "En revisión humana"     -> reembolso, fraude, duda o salida inválida
     rojo   "Bloqueado por política" -> acción no permitida, inyección o promesa de dinero
6. Guarda un borrador en data/cola_analista.json (trazabilidad). Todo reembolso
   queda "pendiente_aprobacion_humana" hasta que una persona lo apruebe.

Es el único componente que escribe, y solo escribe borradores.
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from agent import llm, policy, tools
from agent.schemas import (
    ESTADO_BORRADOR,
    ESTADO_PENDIENTE_APROBACION,
    AccionFinal,
    Chequeo,
    PropuestaAgente,
    ResultadoPolitica,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CASES_PATH = DATA_DIR / "cases.json"
COLA_PATH = DATA_DIR / "cola_analista.json"
MAX_LLAMADAS = 5  # consultas de lectura por turno (proponer_decision no cuenta)
MAX_RONDAS = 4    # idas y vueltas con el modelo por turno
MAX_REINTENTOS_JSON = 1  # si proponer_decision llega inválido, se le pide corregir una vez

ESTADO_RESPONDIDO = "respondido_al_cliente"
ESTADO_APROBADO = "aprobado"
ESTADO_RECHAZADO = "rechazado"
ESTADO_RESPONDIDO_ANALISTA = "respondido_por_analista"

VERDE, AMBAR, ROJO, INFO = "verde", "ambar", "rojo", "info"
TITULOS = {
    VERDE: "Respondido al cliente",
    AMBAR: "En revisión humana",
    ROJO: "Bloqueado por política",
    INFO: "Esperando información del cliente",
}

MSG_REVISION = "Revisamos tu caso y lo pasamos a un analista; te avisaremos aquí."
MSG_PLACA = "Detectamos un posible error en la lectura de tu placa; un analista lo revisa y te avisamos aquí."
MSG_SEGURIDAD = "Por seguridad, un analista revisará tu caso."
MSG_ELEGIR_TX = "¡Claro! ¿Cuál tanqueo quieres revisar? Tócalo en «Tus tanqueos recientes»."
MSG_RECHAZO = "Un analista revisó tu caso y esta vez no procede el reembolso.\nSi tienes más información, escríbenos por aquí."
MSG_RESUELTO = "Un analista revisó tu caso y ya quedó resuelto. ¡Gracias por avisarnos!"


def limpiar_formato(texto: str) -> str:
    """Deja en texto plano la respuesta del LLM: sin markdown y con líneas cortas y limpias."""
    lineas = []
    for linea in texto.replace("\r", "").split("\n"):
        linea = re.sub(r"^\s*(#{1,6}\s+|[-*•]\s+|\d+[.)]\s+)", "", linea)  # títulos, viñetas, listas
        linea = re.sub(r"(\*\*|__|`)", "", linea).strip()                   # negritas y código
        if linea or (lineas and lineas[-1]):                                # máximo una línea en blanco
            lineas.append(linea)
    return "\n".join(lineas).strip()


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cop(valor: Optional[int]) -> str:
    return "—" if valor is None else f"${valor:,.0f}".replace(",", ".")


# --- Casos de demostración ----------------------------------------------------------

def listar_casos() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))["casos"]


def cargar_caso(case_id: str) -> dict:
    for caso in listar_casos():
        if caso["case_id"] == case_id:
            return caso
    raise KeyError(f"Caso no encontrado: {case_id}")


def caso_de_tx(tx_id: str) -> Optional[dict]:
    """El caso de demostración asociado a una transacción (para 'Estás en: Caso N')."""
    for n, caso in enumerate(listar_casos(), 1):
        if caso["tx_id"] == tx_id:
            return {"numero": n, "case_id": caso["case_id"], "titulo": caso["titulo"]}
    return None


# --- Intención (solo informativa, para el panel) -------------------------------------

def detectar_intencion(mensaje: str) -> str:
    if policy.detectar_inyeccion(mensaje):
        return "Intenta dar instrucciones al asistente"
    if re.search(r"no\s+(reconozco|recuerdo|fui|estuve|pude\s+haber\s+sido|he\s+sido|hice)|no\s+tanque[ée]\b|un\s+cobro\s+que\s+no",
                 mensaje, re.IGNORECASE):
        return "No reconoce el cobro"
    if re.search(r"de\s+m[aá]s|cobraron\s+\$|monto|valor|abuso|caro", mensaje, re.IGNORECASE):
        return "Cree que le cobraron un monto incorrecto"
    return "Otra consulta"


# --- Herramientas: el LLM pide, el orquestador ejecuta (solo lectura) ------------------

_LECTURA = {
    "get_transaccion": tools.get_transaccion,
    "get_foto_placa": tools.get_foto_placa,
    "get_venta_fcc": tools.get_venta_fcc,
    "get_estado_push": tools.get_estado_push,
    "get_senales_fraude": tools.get_senales_fraude,
}


def _resumen(nombre: str, d: dict) -> str:
    if nombre == "get_transaccion":
        return (f"{d['estacion']}, surtidor {d['surtidor']}, {d['fecha_hora'][:16].replace('T', ' ')} · "
                f"cobrado {cop(d['monto_capturado_cop'])} (hold {cop(d['hold_cop'])})")
    if nombre == "get_foto_placa":
        estado = "coinciden" if d["coincide_con_cuenta"] else "NO coinciden"
        return f"Foto {d['placa_leida']} vs. cuenta {d['placa_cuenta']}: {estado} (OCR {d['confianza_ocr']:.0%})"
    if nombre == "get_venta_fcc":
        cuadra = "cuadra" if d["cuadra_con_captura"] else "NO cuadra"
        return f"{d['galones']:g} gal × {cop(d['precio_galon_cop'])} = {cop(d['monto_venta_cop'])}; {cuadra} con el cobro"
    if nombre == "get_estado_push":
        return "Notificación push entregada" if d["push_entregado"] else "Notificación push NO entregada"
    if nombre == "get_senales_fraude":
        if not d["senales"]:
            return "Sin señales de fraude"
        return f"{len(d['senales'])} señal(es): " + ", ".join(s.split(":")[0] for s in d["senales"])
    return ""


def ejecutar_herramienta(llamada: llm.LlamadaHerramienta, tx_id: str) -> llm.ResultadoHerramienta:
    funcion = _LECTURA.get(llamada.nombre)
    if funcion is None:
        return llm.ResultadoHerramienta(llamada.id, llamada.nombre, error=f"Herramienta no permitida: {llamada.nombre}")
    pedido = str(llamada.entrada.get("tx_id", ""))
    if pedido != tx_id:
        return llm.ResultadoHerramienta(
            llamada.id, llamada.nombre,
            error=f"Solo se puede consultar la transacción del reclamo ({tx_id}); se pidió '{pedido}'.",
        )
    datos = funcion(tx_id)
    return llm.ResultadoHerramienta(llamada.id, llamada.nombre, datos=datos, ruta_imagen=datos.get("ruta_imagen"))


def _ejecutar_sesion(tx_id: str, mensaje: str, historial: list[dict]):
    """Corre el ciclo de herramientas. Devuelve (propuesta|None, consultas, error|None, n_lecturas).

    El límite MAX_LLAMADAS aplica a las consultas de lectura; proponer_decision (la
    salida del turno) siempre se acepta. Una consulta por encima del límite no se
    ejecuta: el modelo recibe un aviso para que decida. Si insiste más allá de
    MAX_RONDAS, el turno va a revisión humana.
    """
    consultas: list[dict] = []
    lecturas = 0
    intentos_invalidos = 0
    resultados = None
    try:
        sesion = llm.iniciar_sesion(historial, tx_id, mensaje)
        for _ in range(MAX_RONDAS + MAX_REINTENTOS_JSON):
            llamadas = sesion.siguiente(resultados)
            if not llamadas:
                return None, consultas, "El agente terminó sin llamar a proponer_decision.", lecturas
            resultados = []
            for llamada in llamadas:
                if llamada.nombre != llm.HERRAMIENTA_FINAL and lecturas >= MAX_LLAMADAS:
                    resultados.append(llm.ResultadoHerramienta(
                        llamada.id, llamada.nombre,
                        error=f"Límite de {MAX_LLAMADAS} consultas alcanzado. Llama ahora a proponer_decision.",
                    ))
                    continue
                if llamada.nombre == llm.HERRAMIENTA_FINAL:
                    try:
                        return PropuestaAgente.model_validate(llamada.entrada), consultas, None, lecturas
                    except ValidationError as e:
                        detalle = "; ".join(
                            f"{'.'.join(str(p) for p in err['loc']) or 'json'}: {err['msg']}" for err in e.errors()
                        )
                        intentos_invalidos += 1
                        if intentos_invalidos > MAX_REINTENTOS_JSON:
                            return None, consultas, f"JSON inválido en proponer_decision: {detalle}", lecturas
                        # Un reintento: el modelo recibe el error exacto y corrige.
                        resultados.append(llm.ResultadoHerramienta(
                            llamada.id, llamada.nombre,
                            error=f"proponer_decision inválido: {detalle}. Vuelve a llamar a proponer_decision "
                                  "con todos los campos completos (incluye respuesta_cliente y al menos una evidencia citada).",
                        ))
                        continue
                lecturas += 1
                r = ejecutar_herramienta(llamada, tx_id)
                consultas.append({
                    "herramienta": llamada.nombre,
                    "ok": r.error is None,
                    "resumen": r.error or _resumen(llamada.nombre, r.datos),
                })
                resultados.append(r)
        return None, consultas, f"El agente no decidió tras {MAX_RONDAS} rondas (límite de {MAX_LLAMADAS} consultas).", lecturas
    except llm.ErrorLLM as e:
        return None, consultas, str(e), lecturas


# --- Resultado del turno: qué ve el cliente y de qué color es el banner ---------------

def _clasificar(propuesta, politica: ResultadoPolitica, chequeos: list[Chequeo], error: Optional[str],
                evidencia) -> tuple:
    """Devuelve (color, motivo, respuesta_visible, mostrar_foto).

    El cliente solo ve texto del LLM en una aclaración permitida (verde). En los demás
    casos ve un mensaje fijo, y nunca la foto ni la placa de otra persona.
    """
    if propuesta is None:
        return AMBAR, f"No hubo una decisión válida del agente: {error}", MSG_REVISION, False

    fallidos = {c.nombre for c in chequeos if not c.aprobado}
    tipos = [a.tipo for a in politica.acciones_finales]
    corregido = " La política corrigió la decisión del agente." if "coherencia_con_evidencia" in fallidos else ""

    if "sin_inyeccion" in fallidos:
        return ROJO, "Intento de inyección: no se ejecuta ninguna instrucción del mensaje del cliente.", MSG_SEGURIDAD, False
    if "acciones_permitidas" in fallidos:
        motivo = next(c.motivo for c in chequeos if c.nombre == "acciones_permitidas")
        return ROJO, motivo, MSG_SEGURIDAD, False

    if not evidencia.foto_placa["coincide_con_cuenta"]:
        total = sum(a.monto_cop for a in politica.acciones_finales if a.tipo == "recomendar_reembolso")
        return (AMBAR, f"Placas distintas (foto ≠ cuenta): reembolso de {cop(total)} pendiente de una persona "
                f"y caso a MLOps.{corregido}", MSG_PLACA, False)
    if "recomendar_reembolso" in tipos:
        total = sum(a.monto_cop for a in politica.acciones_finales if a.tipo == "recomendar_reembolso")
        return AMBAR, f"Reembolso de {cop(total)} recomendado: requiere la aprobación de un analista.{corregido}", MSG_REVISION, False
    if "sin_senales_fraude" in fallidos or politica.causa_final == "fraude_probable":
        return AMBAR, f"Señales de fraude: el caso lo decide un analista, sin reembolso automático.{corregido}", MSG_SEGURIDAD, False
    if politica.escalado:
        return AMBAR, "Escalado por: " + ", ".join(politica.motivos_escalamiento) + f".{corregido}", MSG_REVISION, False

    # Aclaración permitida: el cliente ve la respuesta del LLM, salvo que prometa
    # dinero o exponga datos de otra persona.
    for chequeo in (policy.revisar_respuesta_cliente(propuesta.respuesta_cliente),
                    policy.revisar_privacidad(propuesta.respuesta_cliente, evidencia)):
        chequeos.append(chequeo)
        if not chequeo.aprobado:
            return ROJO, f"Respuesta del agente bloqueada: {chequeo.motivo} Se reemplazó por un mensaje neutral.", MSG_REVISION, False
    return (VERDE, "Aclaración permitida por la política: el agente respondió con la evidencia.",
            limpiar_formato(propuesta.respuesta_cliente), True)


def turno_sin_transaccion(mensaje: str) -> dict:
    """El cliente escribió sin elegir un tanqueo: el asistente le pregunta cuál (sin LLM)."""
    return {
        "turno_id": f"T-{uuid.uuid4().hex[:6].upper()}",
        "tx_id": None,
        "caso": None,
        "mensaje": mensaje,
        "intencion": detectar_intencion(mensaje),
        "evidencia_consultada": [],
        "llamadas_herramientas": 0,
        "decision_llm": None,
        "causa_final": None,
        "error_llm": None,
        "chequeos": [],
        "acciones_finales": [],
        "resultado": {"color": INFO, "titulo": TITULOS[INFO],
                      "motivo": "El cliente no eligió el tanqueo; el asistente le pregunta cuál es."},
        "respuesta_visible": MSG_ELEGIR_TX,
        "foto": None,
        "borrador_id": None,
    }


def procesar_turno(
    tx_id: str,
    mensaje: str,
    historial: Optional[list[dict]] = None,
    conversacion_id: Optional[str] = None,
) -> dict:
    """Un turno del chat del cliente sobre la transacción elegida."""
    if not mensaje or not mensaje.strip():
        raise ValueError("El mensaje del cliente está vacío.")
    mensaje = mensaje.strip()
    historial = historial or []

    evidencia = tools.get_evidencia(tx_id)  # evidencia completa para el motor de políticas
    propuesta, consultas, error, n_llamadas = _ejecutar_sesion(tx_id, mensaje, historial)

    if propuesta is None:
        politica = ResultadoPolitica(
            chequeos=[Chequeo(nombre="salida_llm_valida", aprobado=False, motivo=error)],
            acciones_finales=[AccionFinal(tipo="escalar_analista", estado=ESTADO_BORRADOR)],
            escalado=True,
            motivos_escalamiento=["salida del LLM inválida"],
        )
        chequeos = list(politica.chequeos)
    else:
        politica = policy.evaluar(propuesta, evidencia, mensaje)
        chequeos = [Chequeo(
            nombre="salida_llm_valida", aprobado=True,
            motivo=f"proponer_decision válido tras {n_llamadas} de {MAX_LLAMADAS} consultas de lectura.",
        )] + list(politica.chequeos)

    color, motivo, respuesta_visible, mostrar_foto = _clasificar(propuesta, politica, chequeos, error, evidencia)
    # Privacidad: la foto solo se muestra si es del vehículo del cliente.
    mostrar_foto = mostrar_foto and evidencia.foto_placa["coincide_con_cuenta"]
    acciones = [a.model_dump() for a in politica.acciones_finales]

    # Borrador (trazabilidad). Solo las aclaraciones en verde se responden sin una persona.
    ahora = _ahora()
    borrador = {
        "borrador_id": f"BOR-{uuid.uuid4().hex[:8].upper()}",
        "conversacion_id": conversacion_id,
        "tx_id": tx_id,
        "mensaje_cliente": mensaje,
        "estado": ESTADO_RESPONDIDO if color == VERDE else ESTADO_PENDIENTE_APROBACION,
        "resultado": color,
        "creado_en": ahora,
        "actualizado_en": ahora,
        "modo_llm": llm.modo_actual()["modo"],
        "modelo": llm.modo_actual()["modelo"],
        "version_prompt": llm.VERSION_PROMPT,
        "causa": politica.causa_final or (propuesta.causa if propuesta else "otra"),
        "acciones": acciones,
        "escalado": politica.escalado,
        "motivo_resultado": motivo,
        "respuesta_visible": respuesta_visible,
        "borrador_respuesta_cliente": propuesta.borrador_respuesta_cliente if propuesta else None,
        "decision_llm": propuesta.model_dump() if propuesta else None,
        "chequeos": [c.model_dump() for c in chequeos],
        "evidencia_consultada": consultas,
        "error_llm": error,
        "historial": [{"fecha": ahora, "actor": "agente", "evento": "borrador_creado", "nota": None}],
    }
    cola = _leer_cola()
    cola.append(borrador)
    _escribir_cola(cola)

    return {
        "turno_id": borrador["borrador_id"],
        "tx_id": tx_id,
        "caso": caso_de_tx(tx_id),
        "mensaje": mensaje,
        "intencion": detectar_intencion(mensaje),
        "evidencia_consultada": consultas,
        "llamadas_herramientas": n_llamadas,
        "decision_llm": borrador["decision_llm"],
        "causa_final": borrador["causa"],
        "error_llm": error,
        "chequeos": borrador["chequeos"],
        "acciones_finales": acciones,
        "resultado": {"color": color, "titulo": TITULOS[color], "motivo": motivo},
        "respuesta_visible": respuesta_visible,
        "foto": evidencia.foto_placa["ruta_imagen"] if mostrar_foto else None,
        "borrador_id": borrador["borrador_id"],
    }


def procesar_caso(case_id: str) -> dict:
    """Atajo: procesa el mensaje de un caso de demostración como primer turno del chat."""
    caso = cargar_caso(case_id)
    return procesar_turno(caso["tx_id"], caso["mensaje_cliente"])


# --- Cola de borradores -----------------------------------------------------------------

def _normalizar(b: dict) -> dict:
    """Completa los campos de borradores guardados con la versión anterior (sin borrar nada)."""
    if "resultado" in b:
        return b
    anterior = b.get("decision_final", {})
    acciones = anterior.get("acciones", [])
    b.setdefault("causa", anterior.get("causa", "otra"))
    b.setdefault("acciones", acciones)
    b.setdefault("escalado", anterior.get("escalado", False))
    b.setdefault("mensaje_cliente", "")
    b.setdefault("motivo_resultado", ", ".join(anterior.get("motivos_escalamiento", [])) or "Versión anterior del prototipo.")
    hay_reembolso = any(a["tipo"] == "recomendar_reembolso" for a in acciones)
    b["resultado"] = AMBAR if hay_reembolso or b["escalado"] else VERDE
    return b


def _leer_cola() -> list[dict]:
    if not COLA_PATH.exists():
        return []
    return [_normalizar(b) for b in json.loads(COLA_PATH.read_text(encoding="utf-8"))]


def _escribir_cola(cola: list[dict]) -> None:
    COLA_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = COLA_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cola, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, COLA_PATH)  # escritura atómica: nunca queda un archivo a medias


def listar_borradores(estado: Optional[str] = None) -> list[dict]:
    return [b for b in _leer_cola() if estado is None or b["estado"] == estado]


def obtener_borrador(borrador_id: str) -> dict:
    for b in _leer_cola():
        if b["borrador_id"] == borrador_id:
            return b
    raise KeyError(f"Borrador no encontrado: {borrador_id}")


def monto_reembolso(borrador: dict) -> int:
    return sum(a["monto_cop"] or 0 for a in borrador["acciones"] if a["tipo"] == "recomendar_reembolso")


# --- Acción humana ------------------------------------------------------------------------

def _resolver(borrador_id: str, revisor: str, nuevo_estado: str, nota: Optional[str],
              extra: Optional[dict] = None) -> dict:
    if not revisor or not revisor.strip():
        raise ValueError("Se requiere el nombre del revisor.")
    cola = _leer_cola()
    for b in cola:
        if b["borrador_id"] != borrador_id:
            continue
        if b["estado"] != ESTADO_PENDIENTE_APROBACION:
            raise ValueError(f"El borrador {borrador_id} ya está '{b['estado']}'.")
        ahora = _ahora()
        b["estado"] = nuevo_estado
        b["actualizado_en"] = ahora
        b["revisado_por"] = revisor.strip()
        b["historial"].append({"fecha": ahora, "actor": revisor.strip(), "evento": nuevo_estado, "nota": nota})
        b.update(extra or {})
        _escribir_cola(cola)
        return b
    raise KeyError(f"Borrador no encontrado: {borrador_id}")


def aprobar_borrador(borrador_id: str, revisor: str, nota: Optional[str] = None) -> dict:
    """Una persona aprueba el borrador (p. ej. el reembolso recomendado)."""
    return _resolver(borrador_id, revisor, ESTADO_APROBADO, nota)


def rechazar_borrador(borrador_id: str, revisor: str, motivo: str) -> dict:
    """Una persona rechaza el borrador; el motivo es obligatorio."""
    if not motivo or not motivo.strip():
        raise ValueError("Se requiere el motivo del rechazo.")
    return _resolver(borrador_id, revisor, ESTADO_RECHAZADO, motivo.strip())


def responder_borrador(borrador_id: str, revisor: str, respuesta: str) -> dict:
    """Una persona responde al cliente un caso en revisión que no lleva reembolso (y lo cierra)."""
    if not respuesta or not respuesta.strip():
        raise ValueError("Escribe la respuesta para el cliente.")
    return _resolver(borrador_id, revisor, ESTADO_RESPONDIDO_ANALISTA, None,
                     extra={"respuesta_analista": respuesta.strip()})


def notificacion_para_cliente(borrador: dict) -> Optional[str]:
    """Mensaje que ve el cliente cuando una persona resuelve su caso."""
    if borrador["estado"] == ESTADO_RESPONDIDO_ANALISTA:
        return borrador.get("respuesta_analista")
    if borrador["estado"] == ESTADO_APROBADO:
        monto = monto_reembolso(borrador)
        if monto:
            return f"Tu reembolso de {cop(monto)} fue aprobado."
        return MSG_RESUELTO
    if borrador["estado"] == ESTADO_RECHAZADO:
        return MSG_RECHAZO
    return None
