#!/usr/bin/env python3
"""Analiza patrones de error en los resultados del benchmark."""

import json
import os
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load all detail files
detalles = {}
for f in glob.glob(os.path.join(BASE_DIR, "detalle_*.json")):
    with open(f) as fp:
        data = json.load(fp)
        detalles[data["nombre"]] = data

# Load main results
with open(os.path.join(BASE_DIR, "resultados_aritmetica.json")) as f:
    resultados = json.load(f)

CATEGORIAS = ["suma_mod32", "xor", "rotacion", "and", "combinada"]

analisis = {"modelos_analizados": []}

for modelo in resultados["modelos"]:
    nombre = modelo["nombre"]
    # Check if any category < 80%
    cats_bajo_80 = []
    for cat in CATEGORIAS:
        if cat in modelo["resultados"] and modelo["resultados"][cat]["accuracy"] < 80:
            cats_bajo_80.append(cat)

    if not cats_bajo_80:
        continue

    print(f"\n{'='*60}")
    print(f"Analizando errores: {nombre}")
    print(f"Categorías bajo 80%: {cats_bajo_80}")

    if nombre not in detalles:
        print(f"  Sin archivo de detalle, saltando")
        continue

    detail = detalles[nombre]["detalle"]
    model_analysis = {
        "nombre": nombre,
        "categorias_analizadas": {}
    }

    for cat in cats_bajo_80:
        cat_items = [d for d in detail if d["categoria"] == cat and d.get("respuesta_parsed") is not None]
        errores = [d for d in cat_items if not d["acerto"]]

        if not errores:
            continue

        # Error absoluto
        errores_abs = []
        errores_rel = []
        error_direccion = {"mayor": 0, "menor": 0, "otro": 0}

        for e in errores:
            parsed = e["respuesta_parsed"]
            correct = e["respuesta_correcta"]
            if parsed is not None and correct is not None:
                abs_err = abs(parsed - correct)
                errores_abs.append(abs_err)
                if correct > 0:
                    errores_rel.append(abs_err / correct)
                if parsed > correct:
                    error_direccion["mayor"] += 1
                elif parsed < correct:
                    error_direccion["menor"] += 1
                else:
                    error_direccion["otro"] += 1

        # Error vs magnitude
        magnitudes_error = []
        for e in errores:
            if e.get("respuesta_parsed") is not None:
                correct = e["respuesta_correcta"]
                magnitude = correct.bit_length() if correct > 0 else 0
                magnitudes_error.append(magnitude)

        magnitudes_correcto = []
        correctos = [d for d in cat_items if d["acerto"]]
        for c in correctos:
            correct = c["respuesta_correcta"]
            magnitude = correct.bit_length() if correct > 0 else 0
            magnitudes_correcto.append(magnitude)

        cat_analysis = {
            "total_preguntas": len(cat_items),
            "errores": len(errores),
            "accuracy": round((len(cat_items) - len(errores)) / len(cat_items) * 100, 1) if cat_items else 0,
            "error_absoluto_promedio": round(sum(errores_abs) / len(errores_abs)) if errores_abs else None,
            "error_absoluto_mediano": sorted(errores_abs)[len(errores_abs)//2] if errores_abs else None,
            "error_relativo_promedio": round(sum(errores_rel) / len(errores_rel), 4) if errores_rel else None,
            "direccion_error": error_direccion,
            "bits_promedio_en_errores": round(sum(magnitudes_error) / len(magnitudes_error), 1) if magnitudes_error else None,
            "bits_promedio_en_correctos": round(sum(magnitudes_correcto) / len(magnitudes_correcto), 1) if magnitudes_correcto else None,
            "patron": "",
            "ejemplos_errores": []
        }

        # Detect patterns
        patterns = []
        if error_direccion["mayor"] > error_direccion["menor"] * 2:
            patterns.append("Tiende a responder números MÁS GRANDES que el correcto")
        elif error_direccion["menor"] > error_direccion["mayor"] * 2:
            patterns.append("Tiende a responder números MÁS PEQUEÑOS que el correcto")

        if cat_analysis["bits_promedio_en_errores"] and cat_analysis["bits_promedio_en_correctos"]:
            if cat_analysis["bits_promedio_en_errores"] > cat_analysis["bits_promedio_en_correctos"] + 3:
                patterns.append("Falla más con números grandes (más bits)")

        if cat_analysis["error_relativo_promedio"] and cat_analysis["error_relativo_promedio"] > 0.5:
            patterns.append("Errores grandes (>50% del valor correcto)")
        elif cat_analysis["error_relativo_promedio"] and cat_analysis["error_relativo_promedio"] < 0.1:
            patterns.append("Errores pequeños (<10% del valor correcto) - cerca pero no exacto")

        cat_analysis["patron"] = "; ".join(patterns) if patterns else "Sin patrón claro detectado"

        # Example errors
        for e in errores[:3]:
            cat_analysis["ejemplos_errores"].append({
                "pregunta_id": e["id"],
                "respuesta_modelo": e["respuesta_parsed"],
                "respuesta_correcta": e["respuesta_correcta"],
                "error_absoluto": abs(e["respuesta_parsed"] - e["respuesta_correcta"]) if e["respuesta_parsed"] is not None else None
            })

        model_analysis["categorias_analizadas"][cat] = cat_analysis
        print(f"  {cat}: {len(errores)} errores, patrón: {cat_analysis['patron']}")

    analisis["modelos_analizados"].append(model_analysis)

# Save
out_path = os.path.join(BASE_DIR, "analisis_errores.json")
with open(out_path, "w") as f:
    json.dump(analisis, f, indent=2, ensure_ascii=False)
print(f"\nAnálisis guardado en {out_path}")
