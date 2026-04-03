#!/usr/bin/env python3
"""
Re-test secuencial de TODOS los modelos OpenAI con timeout 120s.
Va uno por uno para no triggerear rate limits.
"""

import json, os, re, sys, time, urllib.request
from datetime import datetime
from pathlib import Path

KEYS = {}
kf = Path.home() / "zanthu" / "scripts" / ".api_keys"
for line in kf.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        KEYS[k.strip()] = v.strip()

OPENAI_KEY = KEYS["OPENAI_API_KEY"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "preguntas_sha256.json")) as f:
    PREGUNTAS = json.load(f)

CATEGORIAS = ["suma_mod32", "xor", "rotacion", "and", "combinada"]

# Models to test — most interesting first
MODELS = [
    ("openai_gpt54_pro",  "GPT-5.4 Pro",  "gpt-5.4-pro",   "responses"),
    ("openai_o4_mini",    "o4-mini",       "o4-mini",        "responses"),
    ("openai_o3_mini",    "o3-mini",       "o3-mini",        "responses"),
    ("openai_gpt54",      "GPT-5.4",       "gpt-5.4",       "responses"),
    ("openai_gpt4o",      "GPT-4o",        "gpt-4o",        "chat"),
    ("openai_gpt41",      "GPT-4.1",       "gpt-4.1",       "chat"),
    ("openai_gpt54_mini", "GPT-5.4 Mini",  "gpt-5.4-mini",  "responses"),
    ("openai_gpt41_mini", "GPT-4.1 Mini",  "gpt-4.1-mini",  "chat"),
    ("openai_gpt54_nano", "GPT-5.4 Nano",  "gpt-5.4-nano",  "responses"),
    ("openai_gpt4o_mini", "GPT-4o Mini",   "gpt-4o-mini",   "chat"),
    ("openai_gpt41_nano", "GPT-4.1 Nano",  "gpt-4.1-nano",  "chat"),
    ("openai_gpt54_codex","GPT-5.4 Codex", "gpt-5.4-codex", "responses"),
]

def call_openai(model, prompt, api_type, timeout=120):
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json", "User-Agent": "Zanthu/1.0"}
    t0 = time.time()
    try:
        if api_type == "responses":
            payload = json.dumps({"model": model, "input": prompt, "max_output_tokens": 200}).encode()
            req = urllib.request.Request("https://api.openai.com/v1/responses", data=payload, headers=headers)
        else:
            payload = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 200, "temperature": 0}).encode()
            req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=payload, headers=headers)

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())

        if api_type == "responses":
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            return content["text"].strip(), (time.time() - t0) * 1000, None
            return None, (time.time() - t0) * 1000, "No text"
        else:
            return data["choices"][0]["message"]["content"].strip(), (time.time() - t0) * 1000, None
    except Exception as e:
        return None, (time.time() - t0) * 1000, str(e)

def extract_number(text):
    if not text: return None
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    clean = text.replace(",", "").replace(" ", "").replace("*", "").strip()
    if clean.isdigit(): return int(clean)
    matches = re.findall(r'\b(\d+)\b', text)
    if matches:
        for m in reversed(matches):
            n = int(m)
            if n > 0: return n
        return int(matches[0])
    return None

all_results = []

for agent_id, name, model, api_type in MODELS:
    # Check if retest already exists and is complete
    retest_file = os.path.join(BASE_DIR, f"retest_{agent_id}.json")
    if os.path.exists(retest_file):
        with open(retest_file) as f:
            existing = json.load(f)
        total_q = sum(r["total"] for r in existing["resultados"].values())
        if total_q >= 45:  # Allow some errors
            print(f"\n[SKIP] {agent_id} ya testeado ({existing['accuracy_total']}%)", flush=True)
            all_results.append(existing)
            continue

    print(f"\n{'='*60}", flush=True)
    print(f"[RETEST 120s] {agent_id} → {name} ({model})", flush=True)
    print(f"{'='*60}", flush=True)

    cat_scores = {c: {"correctas": 0, "total": 0} for c in CATEGORIAS}
    tiempos = []
    detalle = []
    errores_seguidos = 0
    errores_total = 0

    for pregunta in PREGUNTAS:
        cat = pregunta["categoria"]
        correct = pregunta["respuesta_correcta"]

        resp_text, elapsed, error = call_openai(model, pregunta["prompt"], api_type)

        if error:
            errores_total += 1
            errores_seguidos += 1
            cat_scores[cat]["total"] += 1
            err_short = str(error)[:60]
            print(f"  Q{pregunta['id']:2d} [{cat:10s}] ERR: {err_short} ({elapsed:.0f}ms)", flush=True)
            detalle.append({"id": pregunta["id"], "categoria": cat, "acerto": False,
                           "error": str(error)[:200], "tiempo_ms": round(elapsed, 1)})

            # If rate limited, wait longer
            if "429" in str(error):
                print(f"    → Rate limited, esperando 30s...", flush=True)
                time.sleep(30)
            elif "400" in str(error):
                print(f"    → Bad Request, modelo no disponible. Saltando.", flush=True)
                break
            else:
                time.sleep(3)

            if errores_seguidos >= 5:
                print(f"  >>> 5 errores seguidos, saltando modelo", flush=True)
                break
            continue

        errores_seguidos = 0
        tiempos.append(elapsed)
        parsed = extract_number(resp_text)
        acerto = (parsed == correct)
        cat_scores[cat]["total"] += 1
        if acerto: cat_scores[cat]["correctas"] += 1

        status = "✓" if acerto else "✗"
        print(f"  Q{pregunta['id']:2d} [{cat:10s}] {status} modelo={parsed} correct={correct} ({elapsed:.0f}ms)", flush=True)
        detalle.append({"id": pregunta["id"], "categoria": cat,
                       "respuesta_raw": str(resp_text)[:500], "respuesta_parsed": parsed,
                       "respuesta_correcta": correct, "acerto": acerto, "tiempo_ms": round(elapsed, 1)})
        time.sleep(2)  # Gentle rate limit between successful requests

    total_c = sum(s["correctas"] for s in cat_scores.values())
    total_t = sum(s["total"] for s in cat_scores.values())
    acc = round(total_c / total_t * 100, 1) if total_t > 0 else 0

    resultados_cat = {}
    for cat in CATEGORIAS:
        s = cat_scores[cat]
        a = (s["correctas"] / s["total"] * 100) if s["total"] > 0 else 0
        resultados_cat[cat] = {"correctas": s["correctas"], "total": s["total"], "accuracy": round(a, 1)}

    print(f"\n  [{agent_id}] TOTAL: {total_c}/{total_t} = {acc}%", flush=True)
    for cat in CATEGORIAS:
        r = resultados_cat[cat]
        print(f"    {cat:12s}: {r['correctas']}/{r['total']} = {r['accuracy']}%", flush=True)

    model_data = {
        "agent_id": agent_id, "nombre": name, "model": model,
        "timeout": "120s", "fecha": datetime.now().isoformat(),
        "resultados": resultados_cat, "accuracy_total": acc,
        "tiempo_promedio_ms": round(sum(tiempos)/len(tiempos), 1) if tiempos else 0,
        "errores_api": errores_total, "detalle": detalle
    }
    with open(retest_file, "w") as f:
        json.dump(model_data, f, indent=2, ensure_ascii=False)
    all_results.append(model_data)

    # Wait between models
    print(f"  Esperando 10s antes del siguiente modelo...", flush=True)
    time.sleep(10)

# Final summary
print(f"\n\n{'='*60}", flush=True)
print(f"=== RETEST OPENAI COMPLETO ===", flush=True)
print(f"{'='*60}", flush=True)
for r in sorted(all_results, key=lambda x: x.get("accuracy_total", 0), reverse=True):
    name = r.get("nombre", r.get("agent_id"))
    print(f"  {name:20s} → {r['accuracy_total']}%", flush=True)
    for cat in CATEGORIAS:
        if cat in r["resultados"]:
            c = r["resultados"][cat]
            print(f"    {cat:12s}: {c['correctas']}/{c['total']} = {c['accuracy']}%", flush=True)
print("DONE", flush=True)
