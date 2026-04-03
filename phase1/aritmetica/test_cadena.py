#!/usr/bin/env python3
"""Test de encadenamiento: simula una ronda simplificada de SHA-256."""

import json
import os
import random
import re
import time
import requests

random.seed(99)
MOD32 = 2**32
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_keys():
    keys = {}
    for k in ["GROQ_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY", "OPENAI_API_KEY"]:
        v = os.environ.get(k, "")
        if v:
            keys[k] = v
    if len(keys) < 4:
        for path in [os.path.expanduser("~/zanthu/scripts/.api_keys"), os.path.expanduser("~/zanthu/.env")]:
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line and not line.startswith("#"):
                            k, v = line.split("=", 1)
                            if k.strip() not in keys and v.strip():
                                keys[k.strip()] = v.strip()
    return keys

KEYS = load_keys()

def rotr32(val, n):
    return ((val >> n) | (val << (32 - n))) & 0xFFFFFFFF

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

def query_openai_compat(url, api_key, model, prompt, timeout=60):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 200}
    t0 = time.time()
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        return text.strip(), (time.time() - t0) * 1000, None
    except Exception as e:
        return None, (time.time() - t0) * 1000, str(e)

def query_gemini(api_key, prompt, timeout=60):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0, "maxOutputTokens": 200}}
    t0 = time.time()
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return text.strip(), (time.time() - t0) * 1000, None
    except Exception as e:
        return None, (time.time() - t0) * 1000, str(e)

def query_ollama(base_url, model, prompt, timeout=120):
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0, "num_predict": 200}}
    t0 = time.time()
    try:
        r = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        text = r.json().get("response", "")
        return text.strip(), (time.time() - t0) * 1000, None
    except Exception as e:
        return None, (time.time() - t0) * 1000, str(e)

# Generate 10 chain questions
preguntas_cadena = []
for i in range(10):
    a = random.randint(0, MOD32 - 1)
    b = random.randint(0, MOD32 - 1)
    c = random.randint(0, MOD32 - 1)
    n = random.randint(1, 31)

    x = (a + b) % MOD32
    y = x ^ c
    z = rotr32(y, n)

    prompt = f"""Ejecuta estos 3 pasos en orden y dame SOLO el resultado final:
1. Calcula X = ({a} + {b}) mod 4294967296
2. Calcula Y = X XOR {c}
3. Calcula Z = rotar Y a la derecha {n} bits en registro de 32 bits
Resultado Z = ?"""

    preguntas_cadena.append({
        "id": i + 1,
        "a": a, "b": b, "c": c, "n": n,
        "x_correcto": x, "y_correcto": y, "z_correcto": z,
        "prompt": prompt
    })

def run_chain_test():
    # Load main results to find best model
    results_path = os.path.join(BASE_DIR, "resultados_aritmetica.json")
    if not os.path.exists(results_path):
        print("ERROR: No se encontró resultados_aritmetica.json. Ejecuta benchmark_modelos.py primero.")
        return

    with open(results_path) as f:
        resultados = json.load(f)

    # Find models with >80% accuracy in any individual op
    modelos_buenos = []
    for m in resultados["modelos"]:
        if m["accuracy_total"] >= 30:  # Test any model with at least some capability
            modelos_buenos.append(m["nombre"])

    if not modelos_buenos:
        # Test top 3 anyway
        sorted_models = sorted(resultados["modelos"], key=lambda x: x["accuracy_total"], reverse=True)
        modelos_buenos = [m["nombre"] for m in sorted_models[:3]]

    print(f"Modelos a testear en cadena: {modelos_buenos}")

    # Map model names to query functions
    model_fns = {
        "qwen-3-235b-a22b": lambda p: query_openai_compat("https://api.cerebras.ai/v1/chat/completions", KEYS.get("CEREBRAS_API_KEY",""), "qwen-3-235b-a22b", p),
        "llama-3.3-70b-versatile": lambda p: query_openai_compat("https://api.groq.com/openai/v1/chat/completions", KEYS.get("GROQ_API_KEY",""), "llama-3.3-70b-versatile", p),
        "gemini-2.0-flash": lambda p: query_gemini(KEYS.get("GEMINI_API_KEY",""), p),
        "gpt-4o": lambda p: query_openai_compat("https://api.openai.com/v1/chat/completions", KEYS.get("OPENAI_API_KEY",""), "gpt-4o", p),
        "qwen3:14b": lambda p: query_ollama("http://100.85.169.71:11434", "qwen3:14b", p),
        "qwen2.5:7b": lambda p: query_ollama("http://100.85.169.71:11434", "qwen2.5:7b", p),
        "mistral:7b": lambda p: query_ollama("http://100.85.169.71:11434", "mistral:7b", p),
        "codellama": lambda p: query_ollama("http://100.85.169.71:11434", "codellama", p),
        "qwen2.5:1.5b": lambda p: query_ollama("http://localhost:11434", "qwen2.5:1.5b", p),
    }

    chain_results = {
        "fecha": resultados["fecha"],
        "preguntas_cadena": 10,
        "modelos": []
    }

    for nombre in modelos_buenos:
        if nombre not in model_fns:
            continue

        fn = model_fns[nombre]
        print(f"\n{'='*50}")
        print(f"Chain test: {nombre}")

        correctas = 0
        detalles = []

        for q in preguntas_cadena:
            resp_text, elapsed, error = fn(q["prompt"])
            time.sleep(1)

            if error:
                print(f"  Q{q['id']}: ERROR - {error[:60]}")
                detalles.append({"id": q["id"], "error": error[:200], "acerto": False})
                continue

            parsed = extract_number(resp_text)
            acerto = (parsed == q["z_correcto"])
            if acerto:
                correctas += 1

            status = "✓" if acerto else "✗"
            print(f"  Q{q['id']}: {status} modelo={parsed} correct={q['z_correcto']} ({elapsed:.0f}ms)")

            detalles.append({
                "id": q["id"],
                "respuesta_raw": resp_text[:200],
                "respuesta_parsed": parsed,
                "z_correcto": q["z_correcto"],
                "x_correcto": q["x_correcto"],
                "y_correcto": q["y_correcto"],
                "acerto": acerto,
                "tiempo_ms": round(elapsed, 1)
            })

        acc = round(correctas / 10 * 100, 1)
        print(f"  TOTAL: {correctas}/10 = {acc}%")

        chain_results["modelos"].append({
            "nombre": nombre,
            "correctas": correctas,
            "total": 10,
            "accuracy": acc,
            "detalle": detalles
        })

    out_path = os.path.join(BASE_DIR, "resultados_cadena.json")
    with open(out_path, "w") as f:
        json.dump(chain_results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados de cadena guardados en {out_path}")

if __name__ == "__main__":
    run_chain_test()
