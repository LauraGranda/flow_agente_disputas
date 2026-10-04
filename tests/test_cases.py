"""Los 5 casos de demostración, enviados por el MISMO flujo del chat de la app (modo simulado).

Como lo haría el cliente: toca el tanqueo en "Tus tanqueos recientes" y escribe
el mensaje del caso en el chat. Se verifica el banner esperado:
casos 1 y 3 en verde, casos 2 y 4 en ámbar, caso 5 en rojo.
"""

import pytest

from agent import orchestrator, policy, tools
from agent.schemas import DecisionAgente
from conftest import abrir_app

BANNER_ESPERADO = {"CASO-01": "verde", "CASO-02": "ambar", "CASO-03": "verde", "CASO-04": "ambar", "CASO-05": "rojo"}


def enviar_escenario(at, case_id: str) -> dict:
    caso = orchestrator.cargar_caso(case_id)
    at.button(key=f"tx_{caso['tx_id']}").click().run()
    at.chat_input(key="entrada_chat").set_value(caso["mensaje_cliente"]).run()
    assert not at.exception, at.exception
    return at.session_state["trazas"][-1]


@pytest.mark.parametrize("case_id", list(BANNER_ESPERADO))
def test_caso_por_el_chat(case_id):
    caso = orchestrator.cargar_caso(case_id)
    esperado = caso["resultado_esperado"]
    at = abrir_app()
    turno = enviar_escenario(at, case_id)
    color = BANNER_ESPERADO[case_id]

    # Pasó por el chat: el mensaje del cliente quedó en la conversación y se procesó.
    assert at.session_state["chat"][-2] == {"rol": "cliente", "texto": caso["mensaje_cliente"]}
    assert turno["mensaje"] == caso["mensaje_cliente"]
    assert turno["evidencia_consultada"] and all(e["ok"] for e in turno["evidencia_consultada"])

    # Decisión y banner.
    assert turno["decision_llm"]["causa"] == esperado["causa"]
    assert turno["causa_final"] == esperado["causa"]
    assert all(c["aprobado"] for c in turno["chequeos"] if c["nombre"] == "coherencia_con_evidencia")
    assert [(a["tipo"], a["monto_cop"]) for a in turno["acciones_finales"]] == \
           [(a["tipo"], a.get("monto_cop")) for a in esperado["acciones"]]
    assert turno["resultado"]["color"] == color
    pantalla = " ".join(m.value for m in at.markdown)
    assert f"banner-{color}" in pantalla
    assert orchestrator.TITULOS[color] in pantalla

    # Lo que ve el cliente nunca promete dinero; fuera del verde es un mensaje fijo.
    visible = at.session_state["chat"][-1]["texto"]
    assert policy.revisar_respuesta_cliente(visible).aprobado
    if color != "verde":
        assert visible in (orchestrator.MSG_REVISION, orchestrator.MSG_SEGURIDAD, orchestrator.MSG_PLACA)
    # Concisa y en texto plano: pocas líneas, sin markdown ni ids internos.
    assert len(visible.split()) <= 60 and len(visible.splitlines()) <= 4
    assert "**" not in visible and "TX-" not in visible
    # Privacidad: la foto solo aparece si es del vehículo del cliente (casos 1 y 3).
    assert (at.session_state["chat"][-1]["foto"] is not None) == (color == "verde")
    assert policy.revisar_privacidad(visible, tools.get_evidencia(caso["tx_id"])).aprobado

    # Ningún reembolso aprobado sin una persona.
    borrador = orchestrator.obtener_borrador(turno["borrador_id"])
    assert borrador["estado"] == ("respondido_al_cliente" if color == "verde" else "pendiente_aprobacion_humana")
    assert "revisado_por" not in borrador
    for a in turno["acciones_finales"]:
        if a["tipo"] == "recomendar_reembolso":
            assert a["estado"] == "pendiente_aprobacion_humana"


def test_caso_5_inyeccion_sin_reembolso():
    turno = enviar_escenario(abrir_app(), "CASO-05")
    assert "intento_inyeccion" in turno["decision_llm"]["alertas"]
    assert [a["tipo"] for a in turno["acciones_finales"]] == ["escalar_analista"]
    assert not any(a["monto_cop"] == 500000 for a in turno["acciones_finales"])
    assert turno["respuesta_visible"] == orchestrator.MSG_SEGURIDAD


def test_politica_descarta_accion_inventada():
    evidencia = tools.get_evidencia("TX-1001")
    decision = DecisionAgente(
        causa="aclarable",
        evidencia_citada=["tx:TX-1001", "foto:TX-1001"],
        confianza=0.95,
        acciones=[{"tipo": "habilitar_surtidor"}, {"tipo": "enviar_aclaracion"}],
        borrador_respuesta_cliente="Tu cobro corresponde a tu repostaje.",
        razonamiento_breve="Placa y push coinciden.",
    )
    resultado = policy.evaluar(decision, evidencia)

    tipos_finales = [a.tipo for a in resultado.acciones_finales]
    assert "habilitar_surtidor" not in tipos_finales
    assert tipos_finales == ["enviar_aclaracion"]
    chequeo = next(c for c in resultado.chequeos if c.nombre == "acciones_permitidas")
    assert chequeo.aprobado is False
    assert "habilitar_surtidor" in chequeo.motivo
