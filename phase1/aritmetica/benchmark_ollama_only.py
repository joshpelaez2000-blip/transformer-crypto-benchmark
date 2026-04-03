#!/usr/bin/env python3
"""Benchmark solo modelos Ollama (ASUS + local). Sin APIs externas."""

import json, os, re, time, requests, sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "preguntas_sha256.json")) as f:
    PREGUNTAS = json.load(f)

CATEGORIAS = ["suma_mod32", "xor", "rotacion", "and", "combinada"]

def extract_number(text):
    text = text.strip().replace(",", "").replace(" ", "")
    # Remove think tags if present
    if "<think>" in text:
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

def query_ollama(base_url, model, prompt, timeout=90):
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"temperature": 0, "num_predict": 80}}
    t0 = time.time()
    try:
        r = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        text = r.json().get("response", "")
        return text.strip(), (time.time() - t0) * 1000, None
    except Exception as e:
        return None, (time.time() - t0) * 1000, str(e)

MODELOS = [
    ("qwen3:14b", "ollama_asus", "http://100.85.169.71:11434"),
    ("qwen2.5:7b", "ollama_asus", "http://100.85.169.71:11434"),
    ("mistral:7b", "ollama_asus", "http://100.85.169.71:11434"),
    ("codellama", "ollama_asus", "http://100.85.169.71:11434"),
    ("qwen2.5:1.5b", "ollama_local", "http://localhost:11434"),
]

# Quick connectivity check
for nombre, via, url in MODELOS:
    try:
        r = requests.get(f"{url}/api/tags", timeout=3)
        print(f"  {url}: OK")
        break
    except:
        print(f"  {url}: OFFLINE")

all_results = []

for nombre, via, url in MODELOS:
    print(f"\n{'='*60}", flush=True)
    print(f"Testing: {nombre} via {via}", flush=True)
    print(f"{'='*60}", flush=True)

    cat_scores = {c: {"correctas": 0, "total": 0} for c in CATEGORIAS}
    tiempos = []
    detalle = []
    errores_api = 0

    for i, pregunta in enumerate(PREGUNTAS):
        cat = pregunta["categoria"]
        correct = pregunta["respuesta_correcta"]

        resp_text, elapsed, error = query_ollama(url, nombre, pregunta["prompt"])
        sys.stdout.flush()

        if error:
            print(f"  Q{pregunta['id']}: ERROR - {error[:80]}", flush=True)
            errores_api += 1
            cat_scores[cat]["total"] += 1
            detalle.append({"id": pregunta["id"], "categoria": cat, "respuesta_modelo": None,
                           "respuesta_correcta": correct, "acerto": False, "error_api": error[:200], "tiempo_ms": elapsed})
            if i < 3 and errores_api >= 3:
                print(f"  >>> 3 errores seguidos, saltando", flush=True)
                break
            continue

        tiempos.append(elapsed)
        parsed = extract_number(resp_text)
        acerto = (parsed == correct)
        cat_scores[cat]["total"] += 1
        if acerto:
            cat_scores[cat]["correctas"] += 1

        status = "✓" if acerto else "✗"
        print(f"  Q{pregunta['id']:2d} [{cat:10s}] {status} modelo={parsed} correct={correct} ({elapsed:.0f}ms)", flush=True)

        detalle.append({"id": pregunta["id"], "categoria": cat, "respuesta_raw": resp_text[:300],
                       "respuesta_parsed": parsed, "respuesta_correcta": correct,
                       "acerto": acerto, "tiempo_ms": round(elapsed, 1)})

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

    print(f"\n  TOTAL: {total_c}/{total_t} = {acc_total}%", flush=True)

    model_data = {"nombre": nombre, "via": via, "resultados": resultados_cat,
                  "accuracy_total": acc_total, "tiempo_promedio_ms": tiempo_avg,
                  "errores_api": errores_api, "detalle": detalle}

    # Save detail
    detail_path = os.path.join(BASE_DIR, f"detalle_{nombre.replace(':', '_').replace('.', '_')}.json")
    with open(detail_path, "w") as f:
        json.dump(model_data, f, indent=2, ensure_ascii=False)

    all_results.append({k: v for k, v in model_data.items() if k != "detalle"})

# Merge with existing results
results_path = os.path.join(BASE_DIR, "resultados_aritmetica.json")
if os.path.exists(results_path):
    with open(results_path) as f:
        existing = json.load(f)
else:
    existing = {"fecha": datetime.now().strftime("%Y-%m-%d"), "preguntas_total": 50, "modelos": []}

existing_names = {m["nombre"] for m in existing["modelos"]}
for r in all_results:
    if r["nombre"] in existing_names:
        existing["modelos"] = [m if m["nombre"] != r["nombre"] else r for m in existing["modelos"]]
    else:
        existing["modelos"].append(r)

with open(results_path, "w") as f:
    json.dump(existing, f, indent=2, ensure_ascii=False)

print(f"\n\nResultados Ollama guardados en {results_path}")
print("DONE")
