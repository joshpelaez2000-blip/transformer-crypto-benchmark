# FASE 1: Mapa de Operaciones — Transformer vs SHA-256

## 1. Operaciones exactas de 1 forward pass de GPT-2 (seq_len=1)

| Operación | Cantidad |
|---|---|
| Multiplicaciones float32 | 85,138,944 |
| Sumas float32 | 85,113,406 |
| Exponenciales (exp) | 37,008 |
| Divisiones (div) | 56,402 |
| Raíces cuadradas (sqrt) | 37 |
| **Total FLOPs (mul+sum)** | **170,252,350** |

- Modelo: GPT-2 small (124,439,808 parámetros)
- Verificado: param count manual = PyTorch ✅

## 2. Operaciones exactas de 1 SHA-256 (1 bloque = 512 bits)

| Operación | Por ronda | Total (64 rondas + schedule) |
|---|---|---|
| Sumas mod 2^32 | 7 | 600 |
| Rotaciones (ROTR) | 6 | 576 |
| Shifts (SHR) | 0 | 96 |
| XOR | 7 | 640 |
| AND | 5 | 320 |
| NOT | 1 | 64 |
| **Total** | **26** | **2,296** |

- Verificado contra hashlib: PASS ✅

## 3. Tabla de Equivalencias

| Op SHA-256 | Op Transformer | Existe? | Simulable? | Costo simulación |
|---|---|---|---|---|
| Suma mod 2^32 | Suma float32 | Parcial | Sí | 2 ops float |
| Rotación bits | — | No | Sí | 3 ops float |
| Shift derecho | — | No | Sí | 1 op float |
| XOR 32-bit | — | No | Sí | ~96 ops float |
| AND 32-bit | Mask en attention | Parcial | Sí | ~32 ops float |
| NOT | — | No | Sí | 1 op float |

**Operaciones que el transformer usa y SHA-256 NO tiene:**
- Multiplicación float32 (base de todo el transformer)
- Exponencial (softmax, GELU)
- División float (normalización)
- Raíz cuadrada (LayerNorm, attention scaling)

## 4. Velocidades medidas

| Benchmark | Tiempo | Ops/seg |
|---|---|---|
| GPT-2 forward (1 token, CPU) | **28,514 μs** | 5.97 GFLOPS |
| SHA-256 Python (hashlib) | **0.48 μs** | 2,063,816 hash/s |
| SHA-256 C (OpenSSL, -O3) | **0.42 μs** | 2,379,020 hash/s |

## 5. Conclusión: ¿Cuántos SHA-256 caben en 1 forward pass?

### Por tiempo (misma CPU):
| Comparación | SHA-256 por forward pass |
|---|---|
| vs Python hashlib | **58,848** |
| vs C OpenSSL | **67,843** |

### Por operaciones:
| Métrica | Valor |
|---|---|
| FLOPs por forward pass | 170,252,350 |
| Ops por SHA-256 | 2,296 |
| **Ratio operaciones** | **74,152** |

### Por operaciones (con costo de simulación):
Si el transformer tuviera que *simular* SHA-256 con sus operaciones float:
| Métrica | Valor |
|---|---|
| Ops float para simular 1 SHA-256 | ~74,768 |
| FLOPs disponibles por forward pass | 170,252,350 |
| **SHA-256 simulables por forward pass** | **~2,277** |

### Resumen final

| Métrica | SHA-256 por forward pass |
|---|---|
| Por tiempo (Python) | 58,848 |
| Por tiempo (C) | 67,843 |
| Por conteo de operaciones nativas | 74,152 |
| Por simulación float (costo real) | ~2,277 |

**En el tiempo que GPT-2 procesa 1 token, se podrían computar ~60,000-74,000 SHA-256 con las mismas operaciones nativas, o ~2,277 si el transformer tuviera que simularlos usando sus propias operaciones float32.**
