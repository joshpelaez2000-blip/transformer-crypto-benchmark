# FASE 2: Grokking — Transformer aprende XOR y suma modular

## 1. ¿El transformer aprendió suma modular?

**Sí.** Grokking en epoch 1390, alcanzó 99.7% test accuracy.

| Métrica | Valor |
|---|---|
| Epochs hasta grokking | 1,390 |
| Epochs totales | 1,490 |
| Test accuracy final | 99.7% |
| Tiempo de entrenamiento | 539s (~9 min, CPU) |

## 2. ¿El transformer aprendió XOR?

**Sí.** Grokking en epoch 890 — más rápido que suma modular.

| Métrica | Valor |
|---|---|
| Epochs hasta grokking | 890 |
| Epochs totales | 1,140 |
| Test accuracy final | 99.0% |
| Tiempo de entrenamiento | 423s (~7 min, CPU) |

Resultado inesperado: XOR grokkea MÁS RÁPIDO que la suma modular (890 vs 1390 epochs). Esto sugiere que XOR mod 97, a pesar de no ser una operación modular natural, tiene una estructura que el transformer puede descubrir más fácilmente.

## 3. Gráficos

- `grafico_mod_add.png` — Curva clásica de grokking: train 100% temprano, test salta de ~5% a 99% en epoch ~1000-1390
- `grafico_mod_xor.png` — Mismo patrón, grokking en epoch ~800-890

## 4. Velocidad comparativa

| Método | Tiempo por operación | Ops/segundo | Ratio vs C |
|---|---|---|---|
| **C compilado (-O3)** | 0.86 ns | 1,156,302,964 | 1x |
| **Python nativo** | 75.6 ns | 13,234,184 | 87x más lento |
| **Transformer (single)** | 702 μs | 1,424 | 780,111x más lento |
| **Transformer (batch 1000)** | 24.6 μs/op | 40,638 | 28,455x más lento |

## 5. Modelo

- Arquitectura: 2 capas transformer, 1 head, 128 dims, 512 FFN
- Parámetros: 431,616
- Normalización: RMSNorm
- Activación: SiLU
- Optimizer: AdamW (lr=1e-3, weight_decay=1.0)

## 6. Conclusión

**¿Es práctico usar un transformer para aritmética?**

No, si el objetivo es velocidad. Un transformer entrenado es 780,000x más lento que C para la misma operación. Incluso en batch de 1000, es 28,000x más lento.

Pero ese no es el punto. El resultado importante es que un transformer de 2 capas con 431K parámetros PUEDE aprender XOR modular con 99% accuracy via grokking. Esto demuestra que la operación más problemática de SHA-256 (XOR) es learnable por un transformer, aunque la arquitectura float32 no tiene XOR nativo.

La pregunta de investigación real no es "¿es rápido?" sino "¿es posible?". Y la respuesta es sí.
