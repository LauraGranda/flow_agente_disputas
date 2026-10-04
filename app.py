"""AndesFuel · Asistente Flow — experiencia de cliente tipo chat.

- Columna izquierda: el chat del cliente (lo que ve el cliente).
- Columna derecha: "Detrás de cámaras", el recorrido de cada mensaje
  (intención, evidencia, decisión del LLM, motor de políticas y resultado).
- Pestaña "Panel del analista": la persona que aprueba o rechaza reembolsos y
  responde los casos en revisión que no llevan reembolso.

La UI no habla con el LLM ni escribe en sistemas: cada mensaje pasa por
agent.orchestrator.procesar_turno. Lo único que escribe una persona desde aquí
es la aprobación o el rechazo de un borrador.

Ejecutar: streamlit run app.py
"""

import uuid
from html import escape

import pandas as pd
import streamlit as st

from agent import orchestrator, tools

VERDE = "#6DFD8C"
VERDE_OSCURO = "#15601D"
GRIS = "#212121"

BIENVENIDA = "¡Hola! Bienvenido a AndesFuel 👋 Soy tu asistente de Flow. Cuéntanos en qué podemos ayudarte."
INICIO_RAPIDO = ["No reconozco un cobro", "Me cobraron de más", "Otra consulta"]

ICONO_RESULTADO = {"verde": "🟢", "ambar": "🟠", "rojo": "🔴", "info": "💬"}
NOMBRES_CAUSA = {
    "aclarable": "Aclarable",
    "error_vinculacion_placa": "Error de vinculación de placa",
    "monto": "Monto (retención vs. cobro)",
    "fraude_probable": "Fraude probable",
    "otra": "Otra",
}
NOMBRES_ACCION = {
    "enviar_aclaracion": "✉️ Enviar aclaración",
    "recomendar_reembolso": "💸 Recomendar reembolso",
    "escalar_analista": "🧑‍💼 Escalar a analista",
    "abrir_caso_mlops": "🛠️ Abrir caso MLOps",
}
NOMBRES_ESTADO = {
    "pendiente_aprobacion_humana": "Pendiente de aprobación humana",
    "respondido_al_cliente": "Respondido al cliente",
    "respondido_por_analista": "Respondido por analista",
    "borrador": "Borrador",
    "aprobado": "Aprobado",
    "rechazado": "Rechazado",
}

st.set_page_config(page_title="AndesFuel · Asistente Flow", page_icon="⛽", layout="wide")

st.markdown(
    f"""
<style>
  h1, h2, h3, h4 {{ color: {VERDE_OSCURO} !important; }}
  .block-container {{ padding-top: 3.5rem; }}
  /* "Detrás de cámaras" acompaña el scroll mientras el chat crece */
  [data-testid="stColumn"]:has(.st-key-detras) {{ position: sticky; top: 3.5rem; align-self: flex-start; }}
  .stButton button[kind="primary"], .stButton button[kind="primary"] p {{ color: {GRIS} !important; font-weight: 600; }}
  [data-testid="stTab"][aria-selected="true"],
  [data-testid="stTab"][aria-selected="true"] p {{ color: {VERDE_OSCURO} !important; font-weight: 700; }}

  .logo {{ font-size: 2rem; font-weight: 800; color: {VERDE_OSCURO}; letter-spacing: -0.02em; line-height: 1.1; }}
  .logo .marca {{ border-bottom: 4px solid {VERDE}; }}
  .logo .producto {{ font-weight: 500; color: {GRIS}; }}

  .chip {{ display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px; font-size: 0.85rem;
           font-weight: 600; margin: 0 0.35rem 0.35rem 0; border: 1px solid transparent; }}
  .chip-verde {{ background: {VERDE}; color: {GRIS}; }}
  .chip-oscuro {{ background: {VERDE_OSCURO}; color: #FFFFFF; }}
  .chip-gris {{ background: #ECEFED; color: {GRIS}; border-color: #D5DAD6; }}
  .chip-alerta {{ background: #FDECEA; color: #8A1C12; border-color: #F5C2BC; }}

  /* Burbujas del chat: cliente a la derecha, agente a la izquierda */
  .fila {{ display: flex; align-items: flex-end; gap: 0.5rem; margin: 0.45rem 0; }}
  .fila-cliente {{ justify-content: flex-end; }}
  .avatar {{ width: 2rem; height: 2rem; border-radius: 50%; background: {VERDE_OSCURO}; color: #fff;
             display: flex; align-items: center; justify-content: center; font-size: 1rem; flex-shrink: 0; }}
  .burbuja {{ max-width: 80%; padding: 0.65rem 0.95rem; border-radius: 18px; line-height: 1.45; color: {GRIS}; }}
  .b-agente {{ background: {VERDE}; border-bottom-left-radius: 4px; }}
  .b-cliente {{ background: #EEF1EF; border-bottom-right-radius: 4px; }}
  .b-aviso {{ background: #FFFFFF; border: 2px solid {VERDE}; }}

  /* Detrás de cámaras */
  .paso {{ border-left: 3px solid #DDE5DF; padding: 0.15rem 0 0.6rem 0.9rem; margin-left: 0.4rem; }}
  .paso-titulo {{ font-weight: 700; color: {VERDE_OSCURO}; margin-bottom: 0.25rem; }}
  .linea {{ font-size: 0.92rem; color: {GRIS}; margin: 0.15rem 0; }}
  .banner {{ border-radius: 14px; padding: 1rem 1.2rem; margin-top: 0.6rem; border: 2px solid; }}
  .banner-titulo {{ font-size: 1.35rem; font-weight: 800; }}
  .banner-motivo {{ font-size: 0.95rem; margin-top: 0.25rem; }}
  .banner-verde {{ background: #E6FDEB; border-color: {VERDE}; color: {VERDE_OSCURO}; }}
  .banner-ambar {{ background: #FFF4E0; border-color: #F5A623; color: #7A4A00; }}
  .banner-rojo {{ background: #FDECEA; border-color: #E5484D; color: #8A1C12; }}
  .banner-info {{ background: #EEF2F0; border-color: #C9D3CC; color: {GRIS}; }}

  .pie {{ text-align: center; color: {VERDE_OSCURO}; font-weight: 600; padding: 1.5rem 0 0.5rem;
          border-top: 2px solid {VERDE}; margin-top: 2rem; }}
</style>
""",
    unsafe_allow_html=True,
)


# --- Utilidades de presentación ------------------------------------------------------------

def html(texto: str) -> str:
    """Escapa texto dinámico para HTML. El $ se escapa para que Streamlit no lo lea como LaTeX."""
    return escape(str(texto)).replace("$", "&#36;").replace("\n", "<br>")


def md(texto: str) -> str:
    """Escapa el $ en texto Markdown (captions, avisos)."""
    return texto.replace("$", r"\$")


def chip(texto: str, clase: str = "chip-gris") -> str:
    return f'<span class="chip {clase}">{html(texto)}</span>'


cop = orchestrator.cop


def fecha_corta(iso: str) -> str:
    meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    return f"{int(iso[8:10])} {meses[int(iso[5:7]) - 1]} · {iso[11:16]}"


# --- Estado de la sesión -----------------------------------------------------------------------

ss = st.session_state


def nueva_conversacion() -> None:
    ss.chat = [{"rol": "agente", "texto": BIENVENIDA, "foto": None}]
    ss.trazas = []
    ss.tx_sel = None
    ss.por_procesar = None      # índice en ss.chat del mensaje del cliente que falta procesar
    ss.pendiente_sin_tx = None  # mensaje que espera a que el cliente elija el tanqueo
    ss.mis_borradores = []
    ss.notificados = []
    ss.conversacion_id = f"CONV-{uuid.uuid4().hex[:6].upper()}"


if "chat" not in ss:
    nueva_conversacion()
ss.setdefault("aviso_analista", None)


# --- Callbacks (se ejecutan antes de dibujar la página) ------------------------------------------

def enviar(texto: str) -> None:
    texto = (texto or "").strip()
    if texto:
        ss.chat.append({"rol": "cliente", "texto": texto})
        ss.por_procesar = len(ss.chat) - 1


def cb_entrada_chat() -> None:
    enviar(ss.get("entrada_chat"))


def cb_elegir_tx(tx_id: str) -> None:
    ss.tx_sel = tx_id
    if ss.pendiente_sin_tx is not None:  # el agente había preguntado cuál tanqueo: se retoma
        ss.por_procesar, ss.pendiente_sin_tx = ss.pendiente_sin_tx, None


def cb_responder(borrador_id: str) -> None:
    try:
        orchestrator.responder_borrador(borrador_id, ss.get("revisor", ""), ss.get(f"respuesta_{borrador_id}", ""))
        ss.aviso_analista = None
    except ValueError as e:
        ss.aviso_analista = f"{borrador_id}: {e}"


def cb_resolver(borrador_id: str, aprobar: bool) -> None:
    try:
        if aprobar:
            orchestrator.aprobar_borrador(borrador_id, ss.get("revisor", ""))
        else:
            orchestrator.rechazar_borrador(borrador_id, ss.get("revisor", ""), ss.get(f"motivo_{borrador_id}", ""))
        ss.aviso_analista = None
    except ValueError as e:
        ss.aviso_analista = f"{borrador_id}: {e}"


# --- Lógica de cada rerun: notificaciones y mensaje pendiente ---------------------------------------

def sincronizar_notificaciones() -> None:
    """Si un analista resolvió un borrador de esta conversación, se avisa en el chat."""
    for borrador_id in ss.mis_borradores:
        if borrador_id in ss.notificados:
            continue
        try:
            texto = orchestrator.notificacion_para_cliente(orchestrator.obtener_borrador(borrador_id))
        except KeyError:
            continue
        if texto:
            ss.chat.append({"rol": "agente", "texto": texto, "foto": None, "aviso": True})
            ss.notificados.append(borrador_id)


def procesar_pendiente() -> None:
    idx, ss.por_procesar = ss.por_procesar, None
    texto = ss.chat[idx]["texto"]
    if ss.tx_sel is None:
        turno = orchestrator.turno_sin_transaccion(texto)
        ss.pendiente_sin_tx = idx
    else:
        historial = [{"rol": m["rol"], "texto": m["texto"]} for m in ss.chat[:idx]]
        turno = orchestrator.procesar_turno(ss.tx_sel, texto, historial, ss.conversacion_id)
        ss.mis_borradores.append(turno["borrador_id"])
    ss.trazas.append(turno)
    ss.chat.append({"rol": "agente", "texto": turno["respuesta_visible"], "foto": turno["foto"]})


sincronizar_notificaciones()
if ss.por_procesar is not None:
    with st.spinner("El asistente está revisando tu caso…"):
        procesar_pendiente()

transacciones = tools.listar_transacciones()

# --- Encabezado ------------------------------------------------------------------------------------
st.markdown('<div class="logo"><span class="marca">AndesFuel</span> <span class="producto">· Asistente Flow</span></div>',
            unsafe_allow_html=True)
st.caption("Repostaje sin fricción · reclamos de “no reconozco este cobro” o “me cobraron de más”")

tab_cliente, tab_analista = st.tabs(["Asistente Flow", "Panel del analista"])

# --- Pestaña del cliente -------------------------------------------------------------------------------
with tab_cliente:
    col_chat, col_detras = st.columns([3, 2], gap="large")

    with col_chat:
        caso = orchestrator.caso_de_tx(ss.tx_sel) if ss.tx_sel else None
        st.markdown(
            chip(f"Estás en: Caso {caso['numero']} · {caso['titulo']}", "chip-oscuro") if caso
            else chip("Estás en: elige un tanqueo para empezar", "chip-gris"),
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown("**Tus tanqueos recientes**")
            for t in transacciones:
                st.button(
                    f"⛽ {t['estacion']} · {fecha_corta(t['fecha_hora'])} · {cop(t['monto_capturado_cop'])}",
                    key=f"tx_{t['tx_id']}", on_click=cb_elegir_tx, args=(t["tx_id"],), width="stretch",
                    type="primary" if t["tx_id"] == ss.tx_sel else "secondary",
                )

        for i, m in enumerate(ss.chat):
            if m["rol"] == "cliente":
                st.markdown(f'<div class="fila fila-cliente"><div class="burbuja b-cliente">{html(m["texto"])}</div></div>',
                            unsafe_allow_html=True)
                continue
            clase = "b-aviso" if m.get("aviso") else "b-agente"
            st.markdown(
                f'<div class="fila"><div class="avatar">⛽</div><div class="burbuja {clase}">{html(m["texto"])}</div></div>',
                unsafe_allow_html=True,
            )
            if m.get("foto"):
                st.image(m["foto"], width=340, caption="Foto de la cámara en la estación")
            if i == 0 and not any(x["rol"] == "cliente" for x in ss.chat):
                b1, b2, b3 = st.columns(3)
                for col, texto in zip((b1, b2, b3), INICIO_RAPIDO):
                    col.button(texto, key=f"rapido_{texto}", on_click=enviar, args=(texto,), width="stretch")

        st.chat_input("Escribe tu mensaje…", key="entrada_chat", on_submit=cb_entrada_chat)
        st.button("🔄 Nueva conversación", key="nueva_conversacion", on_click=nueva_conversacion)

    with col_detras, st.container(key="detras"):
        st.subheader("Detrás de cámaras")
        if not ss.trazas:
            st.info("Escribe un mensaje en el chat para ver aquí, paso a paso, cómo decide el agente.")
        else:
            t = ss.trazas[-1]
            st.caption(md(f"Último mensaje: “{t['mensaje']}”" + (f" · {t['tx_id']}" if t["tx_id"] else "")))

            def paso(titulo: str, cuerpo: str) -> None:
                st.markdown(f'<div class="paso"><div class="paso-titulo">{titulo}</div>{cuerpo}</div>',
                            unsafe_allow_html=True)

            paso("1 · 🎯 Intención detectada", chip(t["intencion"], "chip-oscuro"))

            if t["evidencia_consultada"]:
                lineas = "".join(
                    f'<div class="linea">{"✅" if e["ok"] else "❌"} <b>{html(e["herramienta"])}</b> — {html(e["resumen"])}</div>'
                    for e in t["evidencia_consultada"]
                )
                lineas += f'<div class="linea" style="color:#5f6b62">{t["llamadas_herramientas"]} de {orchestrator.MAX_LLAMADAS} consultas de lectura</div>'
            else:
                lineas = '<div class="linea">— Sin consultas: falta saber qué tanqueo revisar.</div>'
            paso("2 · 🔎 Evidencia consultada", lineas)

            d = t["decision_llm"]
            if d:
                cuerpo = chip(NOMBRES_CAUSA.get(d["causa"], d["causa"]), "chip-oscuro")
                cuerpo += chip(f"Confianza {d['confianza']:.0%}", "chip-verde" if d["confianza"] >= 0.7 else "chip-alerta")
                for alerta in d["alertas"]:
                    cuerpo += chip("⚠ " + alerta, "chip-alerta")
                if t["causa_final"] and t["causa_final"] != d["causa"]:
                    cuerpo += (f'<div class="linea">⚖️ La política la corrigió a: '
                               f'<b>{html(NOMBRES_CAUSA.get(t["causa_final"], t["causa_final"]))}</b></div>')
                cuerpo += "".join(
                    f'<div class="linea">Propone: {html(NOMBRES_ACCION.get(a["tipo"], a["tipo"]))}'
                    f'{html(" · " + cop(a["monto_cop"])) if a.get("monto_cop") else ""}</div>'
                    for a in d["acciones"]
                )
            elif t["error_llm"]:
                cuerpo = f'<div class="linea">❌ {html(t["error_llm"])}</div>'
            else:
                cuerpo = '<div class="linea">— El agente aún no decide: espera el tanqueo.</div>'
            paso("3 · 🤖 Decisión del agente", cuerpo)
            if d:
                with st.expander("JSON de proponer_decision"):
                    st.json(d)

            if t["chequeos"]:
                cuerpo = "".join(
                    f'<div class="linea">{"✅" if c["aprobado"] else "❌"} <b>{html(c["nombre"])}</b> — {html(c["motivo"])}</div>'
                    for c in t["chequeos"]
                )
            else:
                cuerpo = '<div class="linea">— No aplica.</div>'
            paso("4 · ⚖️ Motor de políticas", cuerpo)

            r = t["resultado"]
            finales = ", ".join(
                NOMBRES_ACCION.get(a["tipo"], a["tipo"]) + (f" {cop(a['monto_cop'])}" if a["monto_cop"] else "")
                for a in t["acciones_finales"]
            )
            st.markdown(
                f'<div class="banner banner-{r["color"]}">'
                f'<div class="banner-titulo">{ICONO_RESULTADO[r["color"]]} {html(r["titulo"])}</div>'
                f'<div class="banner-motivo">{html(r["motivo"])}</div>'
                + (f'<div class="banner-motivo"><b>Acciones finales:</b> {html(finales)}</div>' if finales else "")
                + "</div>",
                unsafe_allow_html=True,
            )

# --- Panel del analista -----------------------------------------------------------------------------------
with tab_analista:
    st.subheader("Panel del analista")
    st.text_input("Revisor (persona)", value="Analista Flow", key="revisor")
    if ss.aviso_analista:
        st.error(ss.aviso_analista)

    borradores = orchestrator.listar_borradores()
    if not borradores:
        st.info("Todavía no hay borradores. Escribe en el chat del cliente para generar uno.")
    else:
        por_aprobar = [b for b in borradores
                       if b["estado"] == "pendiente_aprobacion_humana" and orchestrator.monto_reembolso(b)]
        st.markdown(f"#### Reembolsos por aprobar ({len(por_aprobar)})")
        if not por_aprobar:
            st.caption("No hay reembolsos pendientes.")
        for b in reversed(por_aprobar):
            with st.container(border=True):
                st.markdown(
                    chip(b["borrador_id"], "chip-oscuro") + chip(b["tx_id"]) + chip(NOMBRES_CAUSA.get(b["causa"], b["causa"]))
                    + f'<div class="linea"><b>Reembolso recomendado: {html(cop(orchestrator.monto_reembolso(b)))}</b></div>'
                    + f'<div class="linea">Cliente: “{html(b["mensaje_cliente"])}”</div>'
                    + f'<div class="linea">Agente: {html(b["decision_llm"]["razonamiento_breve"] if b["decision_llm"] else b["motivo_resultado"])}</div>',
                    unsafe_allow_html=True,
                )
                st.text_input("Motivo (obligatorio para rechazar)", key=f"motivo_{b['borrador_id']}")
                a1, a2, _ = st.columns([1, 1, 3])
                a1.button("Aprobar", type="primary", key=f"aprobar_{b['borrador_id']}", width="stretch",
                          on_click=cb_resolver, args=(b["borrador_id"], True))
                a2.button("Rechazar", key=f"rechazar_{b['borrador_id']}", width="stretch",
                          on_click=cb_resolver, args=(b["borrador_id"], False))

        por_responder = [b for b in borradores
                         if b["estado"] == "pendiente_aprobacion_humana" and not orchestrator.monto_reembolso(b)]
        st.markdown(f"#### Casos por responder ({len(por_responder)})")
        if not por_responder:
            st.caption("No hay casos esperando respuesta.")
        for b in reversed(por_responder):
            bid = b["borrador_id"]
            fallidos = [c["nombre"] for c in b.get("chequeos", []) if not c["aprobado"]]
            with st.container(border=True):
                st.markdown(
                    chip(bid, "chip-oscuro") + chip(b["tx_id"])
                    + chip(f"{ICONO_RESULTADO[b['resultado']]} {orchestrator.TITULOS[b['resultado']]}")
                    + f'<div class="linea">Cliente: “{html(b["mensaje_cliente"])}”</div>'
                    + f'<div class="linea">Por qué está aquí: {html(b["motivo_resultado"])}</div>'
                    + (f'<div class="linea">Chequeos fallidos: {html(", ".join(fallidos))}</div>' if fallidos else ""),
                    unsafe_allow_html=True,
                )
                # La respuesta la escribe la persona; el agente no le sugiere texto.
                st.text_area("Respuesta para el cliente", key=f"respuesta_{bid}", height=90,
                             placeholder="Escribe la respuesta que verá el cliente en el chat…")
                st.button("Enviar respuesta al cliente", type="primary", key=f"responder_{bid}",
                          on_click=cb_responder, args=(bid,))

        st.markdown("#### Cola de borradores")
        filas = [
            {
                "Borrador": b["borrador_id"],
                "Transacción": b["tx_id"],
                "Resultado": f"{ICONO_RESULTADO[b['resultado']]} {orchestrator.TITULOS[b['resultado']]}",
                "Causa": NOMBRES_CAUSA.get(b["causa"], b["causa"]),
                "Acciones": ", ".join(a["tipo"] + (f" ({cop(a['monto_cop'])})" if a["monto_cop"] else "") for a in b["acciones"]),
                "Estado": NOMBRES_ESTADO.get(b["estado"], b["estado"]),
                "Revisado por": b.get("revisado_por", ""),
                "Creado": b["creado_en"].replace("T", " ")[:16],
            }
            for b in reversed(borradores)
        ]
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
        pendientes = sum(b["estado"] == "pendiente_aprobacion_humana" for b in borradores)
        st.caption(f"{len(borradores)} borradores · {pendientes} pendientes de una persona")

st.markdown('<div class="pie">El LLM solo propone. El código valida. Una persona aprueba.</div>', unsafe_allow_html=True)
