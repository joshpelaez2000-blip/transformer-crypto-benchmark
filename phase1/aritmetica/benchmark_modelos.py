#!/usr/bin/env python3
"""Benchmark de operaciones SHA-256 contra múltiples LLMs."""

import json
import os
import re
import time
import requests
from datetime import datetime

# Load API keys from env or file
def load_keys():
    keys = {}
    # Try env first
    for k in ["GROQ_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY", "OPENAI_API_KEY"]:
        v = os.environ.get(k, "")
        if v:
            keys[k] = v
    # Fill missing from .api_keys file
    if len(keys) < 4:
        for path in [os.path.expanduser("~/zanthu/scripts/.api_keys"), os.path.expanduser("~/zanthu/.env")]:
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line and not line.startswith("#"):
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip()
                            if k not in keys and v:
                                keys[k] = v
    return keys

KEYS = load_keys()
print(f"Keys loaded: {list(KEYS.keys())}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "preguntas_sha256.json")) as f:
    PREGUNTAS = json.load(f)

def extract_number(text):
    """Extract the first number from model response."""
    text = text.strip()
    # Try to find a standalone number (possibly with commas)
    # First try: the whole response is a number
    clean = text.replace(",", "").replace(" ", "").strip()
    if clean.isdigit():
        return int(clean)
    # Look for numbers in the text
    matches = re.findall(r'\b(\d[\d,]*\d|\d+)\b', text)
    if matches:
        # Take the last substantial number (often the answer)
        for m in reversed(matches):
            n = int(m.replace(",", ""))
            if n > 0:
                return n
        return int(matches[0].replace(",", ""))
    return None

def query_openai_compat(url, api_key, model, prompt, timeout=60):
    """Query OpenAI-compatible API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 100
    }
    t0 = time.time()
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        elapsed = (time.time() - t0) * 1000
        return text.strip(), elapsed, None
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return None, elapsed, str(e)

def query_gemini(api_key, prompt, timeout=60):
    """Query Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 100}
    }
    t0 = time.time()
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        elapsed = (time.time() - t0) * 1000
        return text.strip(), elapsed, None
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return None, elapsed, str(e)

def query_ollama(base_url, model, prompt, timeout=120):
    """Query Ollama API."""
    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 100}
    }
    t0 = time.time()
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        text = data.get("response", "")
        elapsed = (time.time() - t0) * 1000
        return text.strip(), elapsed, None
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return None, elapsed, str(e)

# Define models
MODELOS = [
    {"nombre": "qwen-3-235b-a22b", "via": "cerebras_api",
     "fn": lambda p: query_openai_compat("https://api.cerebras.ai/v1/chat/completions", KEYS.get("CEREBRAS_API_KEY",""), "qwen-3-235b-a22b", p)},
    {"nombre": "llama-3.3-70b-versatile", "via": "groq_api",
     "fn": lambda p: query_openai_compat("https://api.groq.com/openai/v1/chat/completions", KEYS.get("GROQ_API_KEY",""), "llama-3.3-70b-versatile", p)},
    {"nombre": "gemini-2.0-flash", "via": "gemini_api",
     "fn": lambda p: query_gemini(KEYS.get("GEMINI_API_KEY",""), p)},
    {"nombre": "gpt-4o", "via": "openai_api",
     "fn": lambda p: query_openai_compat("https://api.openai.com/v1/chat/completions", KEYS.get("OPENAI_API_KEY",""), "gpt-4o", p)},
    {"nombre": "qwen3:14b", "via": "ollama_asus",
     "fn": lambda p: query_ollama("http://100.85.169.71:11434", "qwen3:14b", p)},
    {"nombre": "qwen2.5:7b", "via": "ollama_asus",
     "fn": lambda p: query_ollama("http://100.85.169.71:11434", "qwen2.5:7b", p)},
    {"nombre": "mistral:7b", "via": "ollama_asus",
     "fn": lambda p: query_ollama("http://100.85.169.71:11434", "mistral:7b", p)},
    {"nombre": "codellama", "via": "ollama_asus",
     "fn": lambda p: query_ollama("http://100.85.169.71:11434", "codellama", p)},
    {"nombre": "qwen2.5:1.5b", "via": "ollama_local",
     "fn": lambda p: query_ollama("http://localhost:11434", "qwen2.5:1.5b", p)},
]

CATEGORIAS = ["suma_mod32", "xor", "rotacion", "and", "combinada"]

def run_benchmark():
    resultados = {
        "fecha": datetime.now().strftime("%Y-%m-%d"),
        "preguntas_total": len(PREGUNTAS),
        "modelos": []
    }

    for modelo in MODELOS:
        print(f"\n{'='*60}")
        print(f"Testing: {modelo['nombre']} via {modelo['via']}")
        print(f"{'='*60}")

        model_result = {
            "nombre": modelo["nombre"],
            "via": modelo["via"],
            "resultados": {},
            "detalle": [],
            "errores_api": 0
        }

        tiempos = []
        cat_scores = {c: {"correctas": 0, "total": 0} for c in CATEGORIAS}

        for i, pregunta in enumerate(PREGUNTAS):
            cat = pregunta["categoria"]
            correct = pregunta["respuesta_correcta"]

            resp_text, elapsed, error = modelo["fn"](pregunta["prompt"])
            time.sleep(1)  # Rate limit

            if error:
                print(f"  Q{pregunta['id']}: ERROR - {error[:80]}")
                model_result["errores_api"] += 1
                cat_scores[cat]["total"] += 1
                model_result["detalle"].append({
                    "id": pregunta["id"],
                    "categoria": cat,
                    "respuesta_modelo": None,
                    "respuesta_correcta": correct,
                    "acerto": False,
                    "error_api": error[:200],
                    "tiempo_ms": elapsed
                })
                # If first 3 questions all fail, skip this model
                if i < 3 and model_result["errores_api"] >= 3:
                    print(f"  >>> 3 errores seguidos, saltando modelo")
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
                "id": pregunta["id"],
                "categoria": cat,
                "respuesta_raw": resp_text[:200],
                "respuesta_parsed": parsed,
                "respuesta_correcta": correct,
                "acerto": acerto,
                "tiempo_ms": round(elapsed, 1)
            })

        # Compute per-category results
        total_correct = 0
        total_answered = 0
        for cat in CATEGORIAS:
            s = cat_scores[cat]
            acc = (s["correctas"] / s["total"] * 100) if s["total"] > 0 else 0
            model_result["resultados"][cat] = {
                "correctas": s["correctas"],
                "total": s["total"],
                "accuracy": round(acc, 1)
            }
            total_correct += s["correctas"]
            total_answered += s["total"]

        model_result["accuracy_total"] = round(total_correct / total_answered * 100, 1) if total_answered > 0 else 0
        model_result["tiempo_promedio_ms"] = round(sum(tiempos) / len(tiempos), 1) if tiempos else 0
        # Remove detalle from main result (too verbose), save separately
        resultados["modelos"].append({k: v for k, v in model_result.items() if k != "detalle"})

        print(f"\n  TOTAL: {total_correct}/{total_answered} = {model_result['accuracy_total']}%")

        # Save detail per model
        detail_path = os.path.join(BASE_DIR, f"detalle_{modelo['nombre'].replace(':', '_').replace('.', '_')}.json")
        with open(detail_path, "w") as f:
            json.dump(model_result, f, indent=2, ensure_ascii=False)

    # Save main results
    out_path = os.path.join(BASE_DIR, "resultados_aritmetica.json")
    with open(out_path, "w") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"\n\nResultados guardados en {out_path}")

if __name__ == "__main__":
    run_benchmark()
