"""Motor de políticas determinístico.

Recibe la DecisionAgente (ya validada por Pydantic) y el PaqueteEvidencia, y
devuelve los chequeos aplicados y las acciones finales. No usa LLM.

Reglas:
- Solo se permiten ACCIONES_PERMITIDAS; cualquier otra se descarta.
- Un reembolso nunca supera el monto capturado y siempre queda en estado
  "pendiente_aprobacion_humana". Un reembolso inválido se descarta (no se
  recorta en silencio) y el caso se escala.
- Toda evidencia citada debe existir (y debe citarse al menos una).
- Confianza < 0,7, señales de fraude o alerta de inyección -> escalar_analista.
  Con intento de inyección además se descartan los reembolsos: los decide el analista.
"""

import re
import unicodedata
from typing import Optional

from agent.schemas import (
    ESTADO_BORRADOR,
    ESTADO_PENDIENTE_APROBACION,
    Accion,
    AccionFinal,
    Chequeo,
    DecisionAgente,
    PaqueteEvidencia,
    ResultadoPolitica,
)

ACCIONES_PERMITIDAS = ("enviar_aclaracion", "recomendar_reembolso", "escalar_analista", "abrir_caso_mlops")
UMBRAL_CONFIANZA = 0.7
ALERTA_INYECCION = "intento_inyeccion"

# Segunda línea de defensa: no depender solo de que el LLM marque la alerta.
# Buscan órdenes dirigidas al agente (imperativo o subjuntivo en "tú"), no pedidos
# normales del cliente como "¿me pueden devolver la diferencia?" o "devuélveme mi plata".
# Se aplican sobre el texto en minúsculas y sin tildes (ver _normalizar).
_OLVIDAR = r"(olvida|olvides|olvidar|olvidate|ignora|ignores|ignorar|ignorate|descarta|descartes|omite|omitas)"
_PATRONES_INYECCION = [
    # "olvida todas tus instrucciones", "ignora lo anterior", "que olvides esa información"
    rf"\b{_OLVIDAR}\s+(todo|todas?|tus?|las|los|esa|esas|ese|esos|lo|la|el|cualquier)\b",
    rf"\b{_OLVIDAR}\b.{{0,40}}\b(instrucciones|reglas|informacion|evidencia|politicas?|sistema|anterior)",
    r"\bignore\s+(all|your|previous|the)\b",
    r"\b(system|sistema)\s*:",
    r"sin\s+revision\s+humana",
    r"aprueb\w*\s+(ya|ahora|de\s+inmediato)",
    r"\b(aprueba|apruebame|autoriza|autorizame|ejecuta|procesa)\b.{0,40}\breembols",
    # Órdenes de mover dinero: un reembolso nunca se pide así.
    r"\b(consigna|consignes|consigname|transfiere|transfieras|transfiereme|deposita|deposites|depositame)\b",
    r"\b(reembolsa|reembolses|reembolsame|reenbolsa|devuelveme|devuelvas|abona|abones|abonarme|pagame)\b"
    r".{0,50}\b(a|en)\s+(mi|la|esta|otra|una|tu)\s+cuenta\b",
    r"\b(actua|comportate)\s+como\b",
    r"\b(nuevas?\s+instrucciones|modo\s+(administrador|admin|desarrollador|developer))\b",
    r"\b(desactiva|saltate|salta)\w*\s+(las\s+|tus\s+)?(reglas|politicas|validaciones|revision)",
    r"\bprompt\b",
]


def _normalizar(texto: str) -> str:
    """Minúsculas y sin tildes: 'Información' y 'informacion' cuentan igual."""
    sin_tildes = unicodedata.normalize("NFKD", texto or "")
    return "".join(c for c in sin_tildes if not unicodedata.combining(c)).lower()


def detectar_inyeccion(texto: str) -> bool:
    normal = _normalizar(texto)
    return any(re.search(p, normal) for p in _PATRONES_INYECCION)


# El asistente nunca promete dinero: eso solo lo decide una persona.
# Se bloquean COMPROMISOS ("te devolveremos", "recibirás un reembolso"), no explicaciones
# ("los $87.500 se liberaron", "no corresponde un reembolso").
_DINERO = r"(reembols\w*|devoluci[oó]n|reintegro|abono|tu\s+(dinero|plata))"
_PATRON_PROMESA_DINERO = re.compile(
    "|".join([
        r"\b(reembolsar|devolver|reintegrar|abonar|reversar)(emos|é|e)\b",            # devolveremos, reembolsaré
        r"\bte\s+(reembolsamos|devolvemos|reintegramos|abonamos|pagamos)\b",          # te devolvemos
        r"\b(vamos|voy)\s+a\s+(reembolsar|devolver|reintegrar|abonar|reversar)",      # vamos a devolver
        rf"\b(recibir[aá]s|tendr[aá]s|obtendr[aá]s|te\s+llegar[aá])\s+(un|el|tu|la)?\s*{_DINERO}",
        rf"\b(procederemos|haremos|aplicaremos|gestionaremos|realizaremos)\s+(con\s+)?(el|un|la|tu)\s+{_DINERO}",
        rf"\b{_DINERO}\s+(est[aá]\s+|ya\s+)?(aprobad[oa]|confirmad[oa]|garantizad[oa])",
    ]),
    re.IGNORECASE,
)


def revisar_respuesta_cliente(texto: str) -> Chequeo:
    """Chequea que una respuesta que se mostraría tal cual al cliente no prometa dinero."""
    promete = bool(_PATRON_PROMESA_DINERO.search(texto or ""))
    return Chequeo(
        nombre="respuesta_sin_promesas",
        aprobado=not promete,
        motivo="La respuesta menciona reembolsos o devoluciones; solo una persona puede prometer dinero."
        if promete else "La respuesta al cliente no promete dinero.",
    )


def evaluar(
    decision: DecisionAgente,
    evidencia: PaqueteEvidencia,
    mensaje_cliente: Optional[str] = None,
) -> ResultadoPolitica:
    chequeos: list[Chequeo] = []
    motivos_escalamiento: list[str] = []
    monto_capturado = evidencia.transaccion["monto_capturado_cop"]

    # 1. Lista blanca de acciones.
    permitidas = [a for a in decision.acciones if a.tipo in ACCIONES_PERMITIDAS]
    descartadas = [a.tipo for a in decision.acciones if a.tipo not in ACCIONES_PERMITIDAS]
    chequeos.append(Chequeo(
        nombre="acciones_permitidas",
        aprobado=not descartadas,
        motivo=f"Acciones descartadas por no estar permitidas: {descartadas}" if descartadas
        else "Todas las acciones propuestas están en la lista permitida.",
    ))

    # 1b. Coherencia con la evidencia: el código, no el LLM, tiene la última palabra.
    permitidas, causa_final, correcciones, escalar_por_coherencia = _coherencia_con_evidencia(
        decision.causa, permitidas, evidencia
    )
    chequeos.append(Chequeo(
        nombre="coherencia_con_evidencia",
        aprobado=not correcciones,
        motivo="La política corrigió la decisión del agente: " + "; ".join(correcciones) + "." if correcciones
        else "La decisión concuerda con la evidencia (placas, venta del surtidor y señales de fraude).",
    ))
    if escalar_por_coherencia:
        motivos_escalamiento.append("decisión incoherente con la evidencia")

    # 2. Evidencia citada.
    disponibles = evidencia.ids()
    inexistentes = [e for e in decision.evidencia_citada if e not in disponibles]
    evidencia_ok = bool(decision.evidencia_citada) and not inexistentes
    if not decision.evidencia_citada:
        motivo = "La decisión no cita ninguna evidencia."
    elif inexistentes:
        motivo = f"Evidencia citada que no existe: {inexistentes}"
    else:
        motivo = "Toda la evidencia citada existe."
    chequeos.append(Chequeo(nombre="evidencia_existe", aprobado=evidencia_ok, motivo=motivo))
    if not evidencia_ok:
        motivos_escalamiento.append("evidencia citada inválida")

    # 3. Confianza.
    confianza_ok = decision.confianza >= UMBRAL_CONFIANZA
    chequeos.append(Chequeo(
        nombre="confianza_minima",
        aprobado=confianza_ok,
        motivo=f"Confianza {decision.confianza:.2f} (mínimo {UMBRAL_CONFIANZA}).",
    ))
    if not confianza_ok:
        motivos_escalamiento.append("confianza baja")

    # 4. Señales de fraude en la evidencia.
    senales = evidencia.senales_fraude["senales"]
    chequeos.append(Chequeo(
        nombre="sin_senales_fraude",
        aprobado=not senales,
        motivo=f"Señales de fraude: {senales}" if senales else "No hay señales de fraude.",
    ))
    if senales:
        motivos_escalamiento.append("señales de fraude")

    # 5. Inyección de prompt: alerta del LLM o detección propia sobre el mensaje.
    alerta_llm = ALERTA_INYECCION in decision.alertas
    detectada = bool(mensaje_cliente) and detectar_inyeccion(mensaje_cliente)
    inyeccion = alerta_llm or detectada
    fuentes = [n for n, v in (("alerta del LLM", alerta_llm), ("detector determinístico", detectada)) if v]
    chequeos.append(Chequeo(
        nombre="sin_inyeccion",
        aprobado=not inyeccion,
        motivo=f"Intento de inyección detectado por: {', '.join(fuentes)}." if inyeccion
        else "No se detectó intento de inyección.",
    ))
    if inyeccion:
        motivos_escalamiento.append("intento de inyección")

    # 6. Reembolsos: monto válido y siempre pendientes de aprobación humana.
    finales: list[AccionFinal] = []
    rechazos_reembolso: list[str] = []
    for accion in permitidas:
        if accion.tipo != "recomendar_reembolso":
            finales.append(AccionFinal(tipo=accion.tipo, estado=ESTADO_BORRADOR))
            continue
        if inyeccion:
            rechazos_reembolso.append("descartado por intento de inyección")
        elif accion.monto_cop is None or accion.monto_cop <= 0:
            rechazos_reembolso.append(f"monto inválido ({accion.monto_cop})")
        elif accion.monto_cop > monto_capturado:
            rechazos_reembolso.append(f"{accion.monto_cop} supera el monto capturado {monto_capturado}")
        else:
            finales.append(AccionFinal(
                tipo="recomendar_reembolso",
                monto_cop=accion.monto_cop,
                estado=ESTADO_PENDIENTE_APROBACION,
            ))
    hay_reembolsos = any(a.tipo == "recomendar_reembolso" for a in decision.acciones)
    if hay_reembolsos:
        chequeos.append(Chequeo(
            nombre="reembolso_valido",
            aprobado=not rechazos_reembolso,
            motivo=f"Reembolsos descartados: {rechazos_reembolso}" if rechazos_reembolso
            else f"Reembolso ≤ monto capturado ({monto_capturado}); queda {ESTADO_PENDIENTE_APROBACION}.",
        ))
        if rechazos_reembolso and not inyeccion:
            motivos_escalamiento.append("reembolso inválido")

    # 7. Sin acciones válidas no hay nada que proponer: lo decide una persona.
    if not finales:
        motivos_escalamiento.append("sin acciones válidas")

    # Deduplicar conservando el orden y forzar el escalamiento si corresponde.
    vistos: set[tuple] = set()
    unicas = []
    for a in finales:
        clave = (a.tipo, a.monto_cop)
        if clave not in vistos:
            vistos.add(clave)
            unicas.append(a)
    if motivos_escalamiento and not any(a.tipo == "escalar_analista" for a in unicas):
        unicas.append(AccionFinal(tipo="escalar_analista", estado=ESTADO_BORRADOR))

    return ResultadoPolitica(
        chequeos=chequeos,
        acciones_finales=unicas,
        escalado=any(a.tipo == "escalar_analista" for a in unicas),
        motivos_escalamiento=motivos_escalamiento,
        causa_final=causa_final,
    )


def _coherencia_con_evidencia(causa: str, acciones: list[Accion], evidencia: PaqueteEvidencia):
    """Compara la decisión del LLM con la evidencia y la corrige donde no cuadra.

    Reglas:
    - Placas distintas (foto ≠ cuenta): el cobro es de otro vehículo. Nunca es una
      aclaración, aunque el monto cuadre con el surtidor. Se recomienda el reembolso
      del monto capturado (pendiente de una persona) y se abre caso a MLOps.
    - Señales de fraude: no hay aclaración ni reembolso; lo decide un analista.
    - Placas iguales y la venta del surtidor cuadra: no hay base para un reembolso.
      Si el LLM lo propone, se descarta y se escala.

    Devuelve (acciones corregidas, causa final, correcciones, escalar).
    """
    foto, fcc, tx = evidencia.foto_placa, evidencia.venta_fcc, evidencia.transaccion
    capturado = tx["monto_capturado_cop"]
    senales = evidencia.senales_fraude["senales"]
    correcciones: list[str] = []
    escalar = False
    tipos = [a.tipo for a in acciones]

    if not foto["coincide_con_cuenta"]:
        if causa != "error_vinculacion_placa":
            correcciones.append(
                f"causa '{causa}' → 'error_vinculacion_placa' (la foto muestra otra placa, no la de la cuenta)"
            )
            causa = "error_vinculacion_placa"
        if "enviar_aclaracion" in tipos:
            correcciones.append("se descarta la aclaración: confirmaría un cobro de otro vehículo")
        reembolsos = [a for a in acciones if a.tipo == "recomendar_reembolso"]
        if not reembolsos:
            correcciones.append(f"se agrega el reembolso de {capturado} (el cobro corresponde a otro vehículo)")
        elif any(a.monto_cop != capturado for a in reembolsos):
            correcciones.append(f"el reembolso se ajusta al monto capturado completo ({capturado})")
        if "abrir_caso_mlops" not in tipos:
            correcciones.append("se agrega el caso a MLOps por la lectura de placa")
        resto = [a for a in acciones if a.tipo not in ("enviar_aclaracion", "recomendar_reembolso", "abrir_caso_mlops")]
        acciones = [Accion(tipo="recomendar_reembolso", monto_cop=capturado), Accion(tipo="abrir_caso_mlops")] + resto
        return acciones, causa, correcciones, escalar

    if senales:
        if causa != "fraude_probable":
            correcciones.append(f"causa '{causa}' → 'fraude_probable' (hay señales de fraude)")
            causa = "fraude_probable"
        quitar = [t for t in ("enviar_aclaracion", "recomendar_reembolso") if t in tipos]
        if quitar:
            correcciones.append(f"se descarta {', '.join(quitar)}: con señales de fraude decide un analista")
        acciones = [a for a in acciones if a.tipo not in quitar]
        return acciones, causa, correcciones, True

    if causa == "error_vinculacion_placa":
        correcciones.append("la causa dice error de placa, pero las placas coinciden")
        causa, escalar = "otra", True
    if "abrir_caso_mlops" in tipos:
        correcciones.append("se descarta el caso a MLOps: la lectura de la placa es correcta")
        acciones = [a for a in acciones if a.tipo != "abrir_caso_mlops"]
    if fcc["cuadra_con_captura"] and "recomendar_reembolso" in tipos:
        correcciones.append("se descarta el reembolso: placa correcta y el cobro cuadra con la venta del surtidor")
        acciones = [a for a in acciones if a.tipo != "recomendar_reembolso"]
        escalar = True
    return acciones, causa, correcciones, escalar


# Placas colombianas de carro: tres letras y tres números (ABC123 / ABC 123 / ABC-123).
_PATRON_PLACA = re.compile(r"\b([A-Z]{3})[\s-]?(\d{3})\b")


def revisar_privacidad(texto: str, evidencia: PaqueteEvidencia) -> Chequeo:
    """La respuesta al cliente no puede mostrar placas que no sean la de su cuenta."""
    propia = evidencia.foto_placa["placa_cuenta"]
    ajenas = sorted({a + b for a, b in _PATRON_PLACA.findall((texto or "").upper())} - {propia})
    return Chequeo(
        nombre="respuesta_sin_datos_de_terceros",
        aprobado=not ajenas,
        motivo=f"La respuesta muestra placas que no son de la cuenta: {ajenas}." if ajenas
        else "La respuesta no expone datos de otras personas.",
    )
