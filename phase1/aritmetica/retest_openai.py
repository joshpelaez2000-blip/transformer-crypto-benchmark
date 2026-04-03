#!/usr/bin/env python3
"""
Re-test de modelos OpenAI con timeout extendido (120s).
Recibe el agent_id como argumento.
"""

import json, os, re, sys, time, urllib.request
from datetime import datetime
from pathlib import Path

AGENT_ID = sys.argv[1] if len(sys.argv) > 1 else "openai_gpt54_pro"

# Load keys
KEYS = {}
kf = Path.home() / "zanthu" / "scripts" / ".api_keys"
if kf.exists():
    for line in kf.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            KEYS[k.strip()] = v.strip()

OPENAI_KEY = KEYS.get("OPENAI_API_KEY", "")

# Agent definitions (OpenAI only)
AGENTS = {
    "openai_gpt54_pro":  {"name": "GPT-5.4 Pro",  "model": "gpt-5.4-pro",   "api": "responses"},
    "openai_gpt54":      {"name": "GPT-5.4",       "model": "gpt-5.4",       "api": "responses"},
    "openai_gpt54_mini": {"name": "GPT-5.4 Mini",  "model": "gpt-5.4-mini",  "api": "responses"},
    "openai_gpt54_nano": {"name": "GPT-5.4 Nano",  "model": "gpt-5.4-nano",  "api": "responses"},
    "openai_gpt54_codex":{"name": "GPT-5.4 Codex", "model": "gpt-5.4-codex", "api": "responses"},
    "openai_gpt4o":      {"name": "GPT-4o",        "model": "gpt-4o",        "api": "chat"},
    "openai_gpt4o_mini": {"name": "GPT-4o Mini",   "model": "gpt-4o-mini",   "api": "chat"},
    "openai_gpt41":      {"name": "GPT-4.1",       "model": "gpt-4.1",       "api": "chat"},
    "openai_gpt41_mini": {"name": "GPT-4.1 Mini",  "model": "gpt-4.1-mini",  "api": "chat"},
    "openai_gpt41_nano": {"name": "GPT-4.1 Nano",  "model": "gpt-4.1-nano",  "api": "chat"},
    "openai_o4_mini":    {"name": "o4-mini",        "model": "o4-mini",       "api": "responses"},
    "openai_o3":         {"name": "o3",             "model": "o3",            "api": "responses"},
    "openai_o3_mini":    {"name": "o3-mini",        "model": "o3-mini",       "api": "responses"},
}

agent = AGENTS[AGENT_ID]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "preguntas_sha256.json")) as f:
    PREGUNTAS = json.load(f)

CATEGORIAS = ["suma_mod32", "xor", "rotacion", "and", "combinada"]

def call_openai(model, prompt, api_type, timeout=120):
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json", "User-Agent": "Zanthu/1.0"}
    t0 = time.time()
    try:
        if api_type == "responses":
            payload = json.dumps({
                "model": model,
                "input": prompt,
                "max_output_tokens": 200,
            }).encode()
            req = urllib.request.Request("https://api.openai.com/v1/responses", data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for content in item.get("content", []):
                            if content.get("type") == "output_text":
                                return content["text"].strip(), (time.time() - t0) * 1000, None
                return None, (time.time() - t0) * 1000, "No text in response"
        else:
            payload = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0,
            }).encode()
            req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"].strip(), (time.time() - t0) * 1000, None
    except Exception as e:
        return None, (time.time() - t0) * 1000, str(e)

def extract_number(text):
    if not text: return None
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Remove markdown, code blocks, etc
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'\*\*.*?\*\*', '', text)
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

print(f"\n{'='*60}", flush=True)
print(f"[RETEST 120s] {AGENT_ID} → {agent['name']} ({agent['model']})", flush=True)
print(f"{'='*60}", flush=True)

cat_scores = {c: {"correctas": 0, "total": 0} for c in CATEGORIAS}
tiempos = []
detalle = []
errores = 0

for pregunta in PREGUNTAS:
    cat = pregunta["categoria"]
    correct = pregunta["respuesta_correcta"]

    resp_text, elapsed, error = call_openai(agent["model"], pregunta["prompt"], agent["api"])

    if error:
        errores += 1
        cat_scores[cat]["total"] += 1
        print(f"  Q{pregunta['id']:2d} [{cat:10s}] ERR: {str(error)[:60]} ({elapsed:.0f}ms)", flush=True)
        detalle.append({"id": pregunta["id"], "categoria": cat, "acerto": False,
                       "error": str(error)[:200], "tiempo_ms": round(elapsed, 1)})
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
    detalle.append({"id": pregunta["id"], "categoria": cat,
                   "respuesta_raw": str(resp_text)[:500], "respuesta_parsed": parsed,
                   "respuesta_correcta": correct, "acerto": acerto, "tiempo_ms": round(elapsed, 1)})
    time.sleep(1)

total_c = sum(s["correctas"] for s in cat_scores.values())
total_t = sum(s["total"] for s in cat_scores.values())
acc = round(total_c / total_t * 100, 1) if total_t > 0 else 0

resultados_cat = {}
for cat in CATEGORIAS:
    s = cat_scores[cat]
    a = (s["correctas"] / s["total"] * 100) if s["total"] > 0 else 0
    resultados_cat[cat] = {"correctas": s["correctas"], "total": s["total"], "accuracy": round(a, 1)}

print(f"\n  [{AGENT_ID}] TOTAL: {total_c}/{total_t} = {acc}%", flush=True)
for cat in CATEGORIAS:
    r = resultados_cat[cat]
    print(f"    {cat:12s}: {r['correctas']}/{r['total']} = {r['accuracy']}%", flush=True)

model_data = {
    "agent_id": AGENT_ID, "nombre": agent["name"], "model": agent["model"],
    "timeout": "120s", "fecha": datetime.now().isoformat(),
    "resultados": resultados_cat, "accuracy_total": acc,
    "tiempo_promedio_ms": round(sum(tiempos)/len(tiempos), 1) if tiempos else 0,
    "errores_api": errores, "detalle": detalle
}

out = os.path.join(BASE_DIR, f"retest_{AGENT_ID}.json")
with open(out, "w") as f:
    json.dump(model_data, f, indent=2, ensure_ascii=False)
print(f"  Guardado en {out}", flush=True)
