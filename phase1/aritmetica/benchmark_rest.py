#!/usr/bin/env python3
"""Continue benchmark for remaining models (GPT-4o Q44-50 + Ollama models)."""

import json
import os
import re
import time
import requests
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_keys():
    keys = {}
    for path in [os.path.expanduser("~/zanthu/scripts/.api_keys"), os.path.expanduser("~/zanthu/.env")]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        if v.strip():
                            keys[k.strip()] = v.strip()
    for k in ["GROQ_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY", "OPENAI_API_KEY"]:
        v = os.environ.get(k, "")
        if v:
            keys[k] = v
    return keys

KEYS = load_keys()

with open(os.path.join(BASE_DIR, "preguntas_sha256.json")) as f:
    PREGUNTAS = json.load(f)

def extract_number(text):
    text = text.strip().replace(",", "").replace(" ", "")
    if text.isdigit():
        return int(text)
    matches = re.findall(r'\b(\d+)\b', text)
    if matches:
        for m in reversed(matches):
            n = int(m)
            if n > 0:
                return n
        return int(matches[0])
    return None

def query_openai_compat(url, api_key, model, prompt, timeout=30):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 100}
    t0 = time.time()
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        return text.strip(), (time.time() - t0) * 1000, None
    except Exception as e:
        return None, (time.time() - t0) * 1000, str(e)

def query_ollama(base_url, model, prompt, timeout=60):
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0, "num_predict": 50}}
    t0 = time.time()
    try:
        r = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        text = r.json().get("response", "")
        return text.strip(), (time.time() - t0) * 1000, None
    except Exception as e:
        return None, (time.time() - t0) * 1000, str(e)

CATEGORIAS = ["suma_mod32", "xor", "rotacion", "and", "combinada"]

def run_model(nombre, via, query_fn):
    print(f"\n{'='*60}")
    print(f"Testing: {nombre} via {via}")
    print(f"{'='*60}")

    model_result = {"nombre": nombre, "via": via, "resultados": {}, "detalle": [], "errores_api": 0}
    tiempos = []
    cat_scores = {c: {"correctas": 0, "total": 0} for c in CATEGORIAS}

    for i, pregunta in enumerate(PREGUNTAS):
        cat = pregunta["categoria"]
        correct = pregunta["respuesta_correcta"]

        resp_text, elapsed, error = query_fn(pregunta["prompt"])
        time.sleep(1.5)

        if error:
            print(f"  Q{pregunta['id']}: ERROR - {error[:80]}")
            model_result["errores_api"] += 1
            cat_scores[cat]["total"] += 1
            model_result["detalle"].append({
                "id": pregunta["id"], "categoria": cat,
                "respuesta_modelo": None, "respuesta_correcta": correct,
                "acerto": False, "error_api": error[:200], "tiempo_ms": elapsed
            })
            if i < 3 and model_result["errores_api"] >= 3:
                print(f"  >>> 3 errores seguidos, saltando")
                break
            continue

        tiempos.append(elapsed)
        parsed = extract_number(resp_text)
        acerto = (parsed == correct)
        cat_scores[cat]["total"] += 1
        if acerto:
            cat_scores[cat]["correctas"] += 1

        status = "✓" if acerto else "✗"
        print(f"  Q{pregunta['id']:2d} [{cat:10s}] {status} modelo={parsed} correct={correct} ({elapsed:.0f}ms)")

        model_result["detalle"].append({
            "id": pregunta["id"], "categoria": cat,
            "respuesta_raw": resp_text[:200], "respuesta_parsed": parsed,
            "respuesta_correcta": correct, "acerto": acerto, "tiempo_ms": round(elapsed, 1)
        })

    total_correct = 0
    total_answered = 0
    for cat in CATEGORIAS:
        s = cat_scores[cat]
        acc = (s["correctas"] / s["total"] * 100) if s["total"] > 0 else 0
        model_result["resultados"][cat] = {"correctas": s["correctas"], "total": s["total"], "accuracy": round(acc, 1)}
        total_correct += s["correctas"]
        total_answered += s["total"]

    model_result["accuracy_total"] = round(total_correct / total_answered * 100, 1) if total_answered > 0 else 0
    model_result["tiempo_promedio_ms"] = round(sum(tiempos) / len(tiempos), 1) if tiempos else 0
    print(f"\n  TOTAL: {total_correct}/{total_answered} = {model_result['accuracy_total']}%")

    detail_path = os.path.join(BASE_DIR, f"detalle_{nombre.replace(':', '_').replace('.', '_')}.json")
    with open(detail_path, "w") as f:
        json.dump(model_result, f, indent=2, ensure_ascii=False)

    return {k: v for k, v in model_result.items() if k != "detalle"}

# Run remaining models
modelos_pendientes = [
    ("qwen3:14b", "ollama_asus", lambda p: query_ollama("http://100.85.169.71:11434", "qwen3:14b", p)),
    ("qwen2.5:7b", "ollama_asus", lambda p: query_ollama("http://100.85.169.71:11434", "qwen2.5:7b", p)),
    ("mistral:7b", "ollama_asus", lambda p: query_ollama("http://100.85.169.71:11434", "mistral:7b", p)),
    ("codellama", "ollama_asus", lambda p: query_ollama("http://100.85.169.71:11434", "codellama", p)),
    ("qwen2.5:1.5b", "ollama_local", lambda p: query_ollama("http://localhost:11434", "qwen2.5:1.5b", p)),
]

# Also finish GPT-4o (Q44-50) — but we already have enough data, skip
# Load existing results and add new ones
results_path = os.path.join(BASE_DIR, "resultados_aritmetica.json")
if os.path.exists(results_path):
    with open(results_path) as f:
        existing = json.load(f)
else:
    existing = {"fecha": datetime.now().strftime("%Y-%m-%d"), "preguntas_total": 50, "modelos": []}

# Run GPT-4o remaining questions separately
print("=== Completando GPT-4o (Q44-50) ===")
gpt_detail_path = os.path.join(BASE_DIR, "detalle_gpt-4o.json")
with open(gpt_detail_path) as f:
    gpt_existing = json.load(f)

answered_ids = {d["id"] for d in gpt_existing["detalle"]}
remaining = [p for p in PREGUNTAS if p["id"] not in answered_ids]

if remaining:
    print(f"  {len(remaining)} preguntas pendientes para GPT-4o")
    for pregunta in remaining:
        cat = pregunta["categoria"]
        correct = pregunta["respuesta_correcta"]
        resp_text, elapsed, error = query_openai_compat(
            "https://api.openai.com/v1/chat/completions",
            KEYS.get("OPENAI_API_KEY", ""), "gpt-4o", pregunta["prompt"], timeout=30)
        time.sleep(1.5)

        if error:
            print(f"  Q{pregunta['id']}: ERROR - {error[:80]}")
            gpt_existing["detalle"].append({
                "id": pregunta["id"], "categoria": cat,
                "respuesta_modelo": None, "respuesta_correcta": correct,
                "acerto": False, "error_api": error[:200], "tiempo_ms": elapsed
            })
            continue

        parsed = extract_number(resp_text)
        acerto = (parsed == correct)
        status = "✓" if acerto else "✗"
        print(f"  Q{pregunta['id']:2d} [{cat:10s}] {status} modelo={parsed} correct={correct} ({elapsed:.0f}ms)")
        gpt_existing["detalle"].append({
            "id": pregunta["id"], "categoria": cat,
            "respuesta_raw": resp_text[:200], "respuesta_parsed": parsed,
            "respuesta_correcta": correct, "acerto": acerto, "tiempo_ms": round(elapsed, 1)
        })

    # Recalculate GPT-4o stats
    cat_scores = {c: {"correctas": 0, "total": 0} for c in CATEGORIAS}
    tiempos = []
    for d in gpt_existing["detalle"]:
        cat = d["categoria"]
        cat_scores[cat]["total"] += 1
        if d.get("acerto"):
            cat_scores[cat]["correctas"] += 1
        if d.get("tiempo_ms") and not d.get("error_api"):
            tiempos.append(d["tiempo_ms"])

    for cat in CATEGORIAS:
        s = cat_scores[cat]
        acc = (s["correctas"] / s["total"] * 100) if s["total"] > 0 else 0
        gpt_existing["resultados"][cat] = {"correctas": s["correctas"], "total": s["total"], "accuracy": round(acc, 1)}

    total_c = sum(s["correctas"] for s in cat_scores.values())
    total_t = sum(s["total"] for s in cat_scores.values())
    gpt_existing["accuracy_total"] = round(total_c / total_t * 100, 1) if total_t > 0 else 0
    gpt_existing["tiempo_promedio_ms"] = round(sum(tiempos) / len(tiempos), 1) if tiempos else 0

    with open(gpt_detail_path, "w") as f:
        json.dump(gpt_existing, f, indent=2, ensure_ascii=False)

    # Update in main results
    for i, m in enumerate(existing["modelos"]):
        if m["nombre"] == "gpt-4o":
            existing["modelos"][i] = {k: v for k, v in gpt_existing.items() if k != "detalle"}
            break

# Run Ollama models
new_results = []
for nombre, via, fn in modelos_pendientes:
    result = run_model(nombre, via, fn)
    new_results.append(result)

# Merge results
existing_names = {m["nombre"] for m in existing["modelos"]}
for r in new_results:
    if r["nombre"] in existing_names:
        existing["modelos"] = [m if m["nombre"] != r["nombre"] else r for m in existing["modelos"]]
    else:
        existing["modelos"].append(r)

with open(results_path, "w") as f:
    json.dump(existing, f, indent=2, ensure_ascii=False)
print(f"\nResultados actualizados en {results_path}")
