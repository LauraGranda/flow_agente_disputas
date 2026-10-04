"""Paquete del agente de disputas de Flow.

Módulos:
- schemas: modelos Pydantic (reclamo, evidencia, propuesta del LLM, borrador).
- tools: lectura de evidencia (solo lectura) desde los datos simulados.
- policy: motor de políticas determinístico (lista blanca de acciones).
- llm: cliente de Claude que solo devuelve una propuesta en JSON.
- orchestrator: flujo determinístico que une todo y guarda borradores.
"""
