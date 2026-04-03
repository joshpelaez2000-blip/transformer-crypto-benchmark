#!/usr/bin/env python3
"""Genera RESUMEN.md a partir de los resultados del benchmark."""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load results
with open(os.path.join(BASE_DIR, "resultados_aritmetica.json")) as f:
    resultados = json.load(f)

chain_path = os.path.join(BASE_DIR, "resultados_cadena.json")
chain = None
if os.path.exists(chain_path):
    with open(chain_path) as f:
        chain = json.load(f)

errors_path = os.path.join(BASE_DIR, "analisis_errores.json")
errors = None
if os.path.exists(errors_path):
    with open(errors_path) as f:
        errors = json.load(f)

CATEGORIAS = ["suma_mod32", "xor", "rotacion", "and", "combinada"]
CAT_NAMES = {
    "suma_mod32": "Suma mod 2³²",
    "xor": "XOR",
    "rotacion": "Rotación bits",
    "and": "AND",
    "combinada": "Combinada (suma+XOR)"
}

lines = []
lines.append("# Benchmark: Operaciones SHA-256 en LLMs")
lines.append(f"\n**Fecha:** {resultados['fecha']}")
lines.append(f"**Preguntas:** {resultados['preguntas_total']} (10 por categoría)")
lines.append(f"**Modelos testeados:** {len(resultados['modelos'])}")
lines.append(f"**Operaciones:** Suma modular 32-bit, XOR, Rotación de bits, AND, Combinada")
lines.append("")

# Ranking
lines.append("## Ranking por accuracy total")
lines.append("")
sorted_models = sorted(resultados["modelos"], key=lambda x: x["accuracy_total"], reverse=True)
lines.append("| # | Modelo | Via | Accuracy | Tiempo promedio |")
lines.append("|---|--------|-----|----------|-----------------|")
for i, m in enumerate(sorted_models):
    lines.append(f"| {i+1} | {m['nombre']} | {m['via']} | **{m['accuracy_total']}%** | {m['tiempo_promedio_ms']:.0f}ms |")
lines.append("")

# Per-category table
lines.append("## Accuracy por operación")
lines.append("")
header = "| Modelo |" + "|".join(CAT_NAMES[c] for c in CATEGORIAS) + "|"
lines.append(header)
sep = "|--------|" + "|".join("---" for _ in CATEGORIAS) + "|"
lines.append(sep)
for m in sorted_models:
    row = f"| {m['nombre']} |"
    for cat in CATEGORIAS:
        if cat in m["resultados"]:
            acc = m["resultados"][cat]["accuracy"]
            row += f" {acc}% |"
        else:
            row += " N/A |"
    lines.append(row)
lines.append("")

# Hardest operation
lines.append("## ¿Cuál es la operación más difícil?")
lines.append("")
cat_avg = {}
for cat in CATEGORIAS:
    accs = []
    for m in resultados["modelos"]:
        if cat in m["resultados"] and m["resultados"][cat]["total"] > 0:
            accs.append(m["resultados"][cat]["accuracy"])
    cat_avg[cat] = sum(accs) / len(accs) if accs else 0

sorted_cats = sorted(cat_avg.items(), key=lambda x: x[1])
for cat, avg in sorted_cats:
    lines.append(f"- **{CAT_NAMES[cat]}**: {avg:.1f}% accuracy promedio")
lines.append("")
hardest = sorted_cats[0]
lines.append(f"**Operación más difícil:** {CAT_NAMES[hardest[0]]} ({hardest[1]:.1f}% promedio)")
lines.append("")

# Chain results
lines.append("## Test de encadenamiento (ronda simplificada SHA-256)")
lines.append("")
if chain and chain.get("modelos"):
    lines.append("Tarea: Calcular X=(A+B)mod2³², Y=X⊕C, Z=rotr(Y,n) en 3 pasos encadenados.")
    lines.append("")
    lines.append("| Modelo | Correctas/10 | Accuracy |")
    lines.append("|--------|-------------|----------|")
    for m in chain["modelos"]:
        lines.append(f"| {m['nombre']} | {m['correctas']}/10 | {m['accuracy']}% |")
    lines.append("")
    best_chain = max(chain["modelos"], key=lambda x: x["accuracy"])
    if best_chain["accuracy"] > 0:
        lines.append(f"**Mejor modelo en cadena:** {best_chain['nombre']} ({best_chain['accuracy']}%)")
    else:
        lines.append("**Ningún modelo completó correctamente las operaciones encadenadas.**")
else:
    lines.append("No se ejecutó el test de cadena (ningún modelo superó el umbral).")
lines.append("")

# Error analysis summary
if errors and errors.get("modelos_analizados"):
    lines.append("## Patrones de error detectados")
    lines.append("")
    for m in errors["modelos_analizados"]:
        lines.append(f"### {m['nombre']}")
        for cat, analysis in m.get("categorias_analizadas", {}).items():
            lines.append(f"- **{CAT_NAMES.get(cat, cat)}**: {analysis.get('patron', 'N/A')}")
            if analysis.get("error_absoluto_promedio"):
                lines.append(f"  - Error absoluto promedio: {analysis['error_absoluto_promedio']:,}")
        lines.append("")

# Conclusion
lines.append("## Conclusión: ¿Puede un LLM computar operaciones SHA-256 via tokens?")
lines.append("")

best = sorted_models[0] if sorted_models else None
if best:
    if best["accuracy_total"] >= 80:
        lines.append(f"**SÍ, parcialmente.** El mejor modelo ({best['nombre']}) alcanza {best['accuracy_total']}% en operaciones individuales.")
        if chain and chain.get("modelos"):
            best_c = max(chain["modelos"], key=lambda x: x["accuracy"])
            if best_c["accuracy"] >= 50:
                lines.append(f"En encadenamiento, {best_c['nombre']} logra {best_c['accuracy']}%, sugiriendo que el forward pass contiene capacidad computacional parcial para SHA-256.")
            else:
                lines.append("Sin embargo, el encadenamiento de operaciones degrada significativamente la accuracy, indicando que la capacidad es frágil.")
    elif best["accuracy_total"] >= 40:
        lines.append(f"**Parcialmente.** El mejor modelo ({best['nombre']}) alcanza {best['accuracy_total']}% — demuestra capacidad aritmética emergente pero insuficiente para SHA-256 completo.")
    else:
        lines.append(f"**NO de forma confiable.** El mejor modelo ({best['nombre']}) solo alcanza {best['accuracy_total']}% — los LLMs no pueden ejecutar consistentemente las operaciones básicas de SHA-256.")

lines.append("")
lines.append("### Implicaciones")
lines.append("")
lines.append("- Las operaciones de SHA-256 requieren precisión exacta en aritmética de 32 bits")
lines.append("- Los LLMs procesan números como tokens (substrings), no como valores numéricos nativos")
lines.append("- La capacidad de hacer estas operaciones 'emerge' del entrenamiento pero no es confiable")
lines.append("- Esto confirma que el forward pass de un LLM NO es equivalente a un circuito aritmético — es una aproximación estadística")

out_path = os.path.join(BASE_DIR, "RESUMEN.md")
with open(out_path, "w") as f:
    f.write("\n".join(lines))
print(f"Resumen generado en {out_path}")
