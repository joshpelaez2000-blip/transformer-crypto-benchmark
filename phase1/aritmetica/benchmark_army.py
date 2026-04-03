#!/usr/bin/env python3
"""
Benchmark SHA-256 ops usando TODA la army de Zanthu (70 agentes).
Solo testea modelos tipo "chat" y "reasoning".
Usa blackboard.py directamente.
"""

import json, os, re, sys, time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.expanduser("~/zanthu/proyectos/blackboard"))
from blackboard import query_agent, AGENTS, PROVIDERS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "preguntas_sha256.json")) as f:
    PREGUNTAS = json.load(f)

CATEGORIAS = ["suma_mod32", "xor", "rotacion", "and", "combinada"]

# Filter: only chat/reasoning models
CHAT_AGENTS = {k: v for k, v in AGENTS.items()
               if v.get("type") in ("chat", "reasoning")}

print(f"Total agentes chat/reasoning: {len(CHAT_AGENTS)}", flush=True)
for k, v in CHAT_AGENTS.items():
    print(f"  {k:25s} {v['provider']:12s} {v['model']}", flush=True)

def extract_number(text):
    if not text:
        return None
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    clean = text.replace(",", "").replace(" ", "").strip()
    if clean.isdigit():
        return int(clean)
    matches = re.findall(r'\b(\d+)\b', text)
    if matches:
        for m in reversed(matches):
            n = int(m)
            if n > 0:
                return n
        return int(matches[0])
    return None

def test_one_agent(agent_id):
    """Test one agent on all 50 questions."""
    agent = CHAT_AGENTS[agent_id]
    print(f"\n{'='*60}", flush=True)
    print(f"[{agent_id}] {agent['name']} ({agent['provider']}/{agent['model']})", flush=True)
    print(f"{'='*60}", flush=True)

    cat_scores = {c: {"correctas": 0, "total": 0} for c in CATEGORIAS}
    tiempos = []
    detalle = []
    errores = 0

    for i, pregunta in enumerate(PREGUNTAS):
        cat = pregunta["categoria"]
        correct = pregunta["respuesta_correcta"]

        t0 = time.time()
        try:
            result = query_agent(agent_id, pregunta["prompt"])
            resp_text = result.get("response", "")
            error = None if resp_text else result.get("error", "empty response")
        except Exception as e:
            resp_text = None
            error = str(e)
        elapsed = (time.time() - t0) * 1000

        if error or not resp_text:
            errores += 1
            cat_scores[cat]["total"] += 1
            err_str = str(error)[:100] if error else "empty"
            print(f"  Q{pregunta['id']:2d} [{cat:10s}] ERR: {err_str}", flush=True)
            detalle.append({"id": pregunta["id"], "categoria": cat, "acerto": False,
                           "error": err_str, "tiempo_ms": round(elapsed, 1)})
            # Bail if first 3 all fail
            if i < 3 and errores >= 3:
                print(f"  >>> 3 errores seguidos, saltando", flush=True)
                break
            time.sleep(1)
            continue

        tiempos.append(elapsed)
        parsed = extract_number(resp_text)
        acerto = (parsed == correct)
        cat_scores[cat]["total"] += 1
        if acerto:
            cat_scores[cat]["correctas"] += 1

        status = "✓" if acerto else "✗"
        print(f"  Q{pregunta['id']:2d} [{cat:10s}] {status} modelo={parsed} correct={correct} ({elapsed:.0f}ms)", flush=True)
        detalle.append({"id": pregunta["id"], "categoria": cat,
                       "respuesta_raw": str(resp_text)[:300], "respuesta_parsed": parsed,
                       "respuesta_correcta": correct, "acerto": acerto, "tiempo_ms": round(elapsed, 1)})
        time.sleep(0.5)  # Gentle rate limit

    # Stats
    total_c = sum(s["correctas"] for s in cat_scores.values())
    total_t = sum(s["total"] for s in cat_scores.values())
    acc_total = round(total_c / total_t * 100, 1) if total_t > 0 else 0
    tiempo_avg = round(sum(tiempos) / len(tiempos), 1) if tiempos else 0

    resultados_cat = {}
    for cat in CATEGORIAS:
        s = cat_scores[cat]
        acc = (s["correctas"] / s["total"] * 100) if s["total"] > 0 else 0
        resultados_cat[cat] = {"correctas": s["correctas"], "total": s["total"], "accuracy": round(acc, 1)}

    print(f"\n  [{agent_id}] TOTAL: {total_c}/{total_t} = {acc_total}%", flush=True)

    model_data = {
        "agent_id": agent_id,
        "nombre": agent["name"],
        "provider": agent["provider"],
        "model": agent["model"],
        "resultados": resultados_cat,
        "accuracy_total": acc_total,
        "tiempo_promedio_ms": tiempo_avg,
        "errores_api": errores,
        "detalle": detalle
    }

    # Save individual detail
    safe_name = agent_id.replace("/", "_").replace(":", "_")
    detail_path = os.path.join(BASE_DIR, f"detalle_{safe_name}.json")
    with open(detail_path, "w") as f:
        json.dump(model_data, f, indent=2, ensure_ascii=False)

    return {k: v for k, v in model_data.items() if k != "detalle"}


# === MAIN ===
resultados = {
    "fecha": datetime.now().strftime("%Y-%m-%d"),
    "preguntas_total": 50,
    "agentes_total": len(CHAT_AGENTS),
    "modelos": []
}

# Run sequentially to respect rate limits
for agent_id in CHAT_AGENTS:
    r = test_one_agent(agent_id)
    resultados["modelos"].append(r)
    # Save intermediate results after each model
    with open(os.path.join(BASE_DIR, "resultados_aritmetica.json"), "w") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

# Final summary
print(f"\n\n{'='*60}", flush=True)
print(f"=== BENCHMARK COMPLETO: {len(resultados['modelos'])} modelos ===", flush=True)
print(f"{'='*60}", flush=True)
for m in sorted(resultados["modelos"], key=lambda x: x["accuracy_total"], reverse=True):
    print(f"  {m['nombre']:25s} {m['provider']:12s} → {m['accuracy_total']}%", flush=True)
print("DONE", flush=True)
