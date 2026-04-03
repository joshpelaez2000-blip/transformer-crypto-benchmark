#!/usr/bin/env python3
"""Benchmark final — lee keys de archivo, nombres correctos, timeouts ajustados."""

import json, os, re, time, requests, sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "preguntas_sha256.json")) as f:
    PREGUNTAS = json.load(f)

CATEGORIAS = ["suma_mod32", "xor", "rotacion", "and", "combinada"]

# Load keys directly from file (not shell source)
KEYS = {}
for path in [os.path.expanduser("~/zanthu/scripts/.api_keys"), os.path.expanduser("~/zanthu/.env")]:
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    KEYS[k.strip()] = v.strip()
print(f"Keys: {[k for k in KEYS if 'KEY' in k or 'TOKEN' in k]}", flush=True)

def extract_number(text):
    text = text.strip()
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

def query_api(url, api_key, model, prompt, timeout=30):
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

def query_gemini(api_key, prompt, timeout=30):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0, "maxOutputTokens": 100}}
    t0 = time.time()
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return text.strip(), (time.time() - t0) * 1000, None
    except Exception as e:
        return None, (time.time() - t0) * 1000, str(e)

def query_ollama(base_url, model, prompt, timeout=120):
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0, "num_predict": 80}}
    t0 = time.time()
    try:
        r = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        text = r.json().get("response", "")
        return text.strip(), (time.time() - t0) * 1000, None
    except Exception as e:
        return None, (time.time() - t0) * 1000, str(e)

# Define all models with correct names and keys from file
MODELOS = []

# APIs
if KEYS.get("OPENAI_API_KEY"):
    MODELOS.append(("gpt-4o", "openai_api",
        lambda p: query_api("https://api.openai.com/v1/chat/completions", KEYS["OPENAI_API_KEY"], "gpt-4o", p)))

if KEYS.get("GROQ_API_KEY"):
    MODELOS.append(("llama-3.3-70b-versatile", "groq_api",
        lambda p: query_api("https://api.groq.com/openai/v1/chat/completions", KEYS["GROQ_API_KEY"], "llama-3.3-70b-versatile", p)))

if KEYS.get("GEMINI_API_KEY"):
    MODELOS.append(("gemini-2.0-flash", "gemini_api",
        lambda p: query_gemini(KEYS["GEMINI_API_KEY"], p)))

if KEYS.get("CEREBRAS_API_KEY"):
    MODELOS.append(("llama-3.3-70b", "cerebras_api",
        lambda p: query_api("https://api.cerebras.ai/v1/chat/completions", KEYS["CEREBRAS_API_KEY"], "llama-3.3-70b", p)))

# Ollama ASUS — skip qwen3:14b (too slow, thinking model)
MODELOS.extend([
    ("qwen2.5:3b", "ollama_asus", lambda p: query_ollama("http://100.85.169.71:11434", "qwen2.5:3b", p)),
    ("qwen2.5:latest", "ollama_asus", lambda p: query_ollama("http://100.85.169.71:11434", "qwen2.5:latest", p)),
    ("mistral:latest", "ollama_asus", lambda p: query_ollama("http://100.85.169.71:11434", "mistral:latest", p)),
    ("llama3:latest", "ollama_asus", lambda p: query_ollama("http://100.85.169.71:11434", "llama3:latest", p)),
    ("codellama:latest", "ollama_asus", lambda p: query_ollama("http://100.85.169.71:11434", "codellama:latest", p)),
    ("llama3.2:1b", "ollama_asus", lambda p: query_ollama("http://100.85.169.71:11434", "llama3.2:1b", p)),
])

# Ollama local
MODELOS.append(("qwen2.5:1.5b", "ollama_local", lambda p: query_ollama("http://localhost:11434", "qwen2.5:1.5b", p)))

def run_model(nombre, via, query_fn):
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
        resp_text, elapsed, error = query_fn(pregunta["prompt"])

        if error:
            print(f"  Q{pregunta['id']}: ERROR - {error[:80]}", flush=True)
            errores_api += 1
            cat_scores[cat]["total"] += 1
            detalle.append({"id": pregunta["id"], "categoria": cat, "respuesta_modelo": None,
                           "respuesta_correcta": correct, "acerto": False, "error_api": error[:200], "tiempo_ms": elapsed})
            if i < 3 and errores_api >= 3:
                print(f"  >>> Saltando modelo (3 errores)", flush=True)
                break
            time.sleep(2)
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
        time.sleep(1)  # Rate limit

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
                  "accuracy_total": acc_total, "tiempo_promedio_ms": tiempo_avg, "detalle": detalle}

    detail_path = os.path.join(BASE_DIR, f"detalle_{nombre.replace(':', '_').replace('.', '_')}.json")
    with open(detail_path, "w") as f:
        json.dump(model_data, f, indent=2, ensure_ascii=False)

    return {k: v for k, v in model_data.items() if k != "detalle"}

# Run all
resultados = {"fecha": datetime.now().strftime("%Y-%m-%d"), "preguntas_total": 50, "modelos": []}
for nombre, via, fn in MODELOS:
    r = run_model(nombre, via, fn)
    resultados["modelos"].append(r)

with open(os.path.join(BASE_DIR, "resultados_aritmetica.json"), "w") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print(f"\n\n=== BENCHMARK COMPLETO ===")
print(f"Modelos testeados: {len(resultados['modelos'])}")
for m in sorted(resultados["modelos"], key=lambda x: x["accuracy_total"], reverse=True):
    print(f"  {m['nombre']:25s} {m['via']:15s} → {m['accuracy_total']}%")
print("DONE", flush=True)
