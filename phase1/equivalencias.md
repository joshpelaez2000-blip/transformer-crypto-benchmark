# Tabla de Equivalencias: SHA-256 vs Transformer (GPT-2)

## Operaciones SHA-256 → Operaciones Transformer

| Op SHA-256 | Cantidad/bloque | Op Transformer equivalente | Dónde ocurre | Diferencia | Simulable? |
|---|---|---|---|---|---|
| **Suma mod 2^32** (ADD32) | 600 | Suma float32 | Residual connections, bias add, LayerNorm (mean, var) | SHA: entero 32-bit con overflow wrap. Transformer: float32 IEEE 754, sin wrap, con decimales | **Parcial** — se puede simular con `x.int() % 2^32` pero pierde la ventaja de float |
| **Rotación bits** (ROTR) | 576 | **No existe** | — | SHA: rotación circular de bits en word 32-bit. Transformer: no tiene operaciones bitwise | **Sí** — se puede simular con: `(x >> n) \| (x << (32-n))` usando multiplicación por potencias de 2, pero es costoso (~3 ops float por 1 ROTR) |
| **Shift derecho** (SHR) | 96 | **No existe** | — | SHA: shift lógico a la derecha. Transformer: no opera a nivel de bits | **Sí** — divisón entera por potencia de 2: `floor(x / 2^n)`, 1 op float |
| **XOR** | 640 | **No existe directo** | — | SHA: XOR bit a bit, reversible. Transformer: no tiene XOR nativo | **Sí** — `a + b - 2*a*b` para bits individuales, pero para words 32-bit requiere descomponer en 32 bits → 32*(add+mul+sub) = ~96 ops float por 1 XOR32 |
| **AND** | 320 | **Parcial** — Attention mask | Attention (masking de tokens futuros) | SHA: AND bit a bit entre words. Transformer: AND solo como mask binario (0/1), no como operación aritmética entre words | **Sí** — `a * b` para bits individuales. Para words 32-bit: descomponer en bits → 32 muls = 32 ops float |
| **NOT** | 64 | **No existe** | — | SHA: inversión de todos los bits. Transformer: no tiene complemento | **Sí** — `0xFFFFFFFF - x` = 1 resta, o bit a bit: `1 - bit` × 32 |

## Análisis de compatibilidad

### Operaciones que el transformer tiene y SHA-256 NO usa:
| Op Transformer | SHA-256 equivalente | Nota |
|---|---|---|
| Multiplicación float32 | No existe | SHA-256 no multiplica. El transformer depende fundamentalmente de matmuls |
| Exponencial (exp) | No existe | Usado en softmax y GELU. SHA-256 es puramente lógico/aritmético entero |
| División float32 | No existe | SHA-256 solo divide implícitamente vía shifts (÷ potencias de 2) |
| Raíz cuadrada | No existe | Usado en LayerNorm y scaling de attention |
| Softmax | No existe | Normalización probabilística. SHA-256 no normaliza nada |

### Resumen de simulabilidad

Para simular **1 bloque SHA-256** usando operaciones de transformer (float32):

| Op SHA-256 | Cantidad | Costo en ops float32 | Total float ops |
|---|---|---|---|
| ADD32 | 600 | ~1 (suma + mod) → 2 ops | 1,200 |
| ROTR | 576 | ~3 ops (shift + shift + or) | 1,728 |
| SHR | 96 | ~1 op (div por 2^n) | 96 |
| XOR (32-bit) | 640 | ~96 ops (bit decompose) | 61,440 |
| AND (32-bit) | 320 | ~32 ops (bit multiply) | 10,240 |
| NOT | 64 | ~1 op (subtract from mask) | 64 |
| **TOTAL** | **2,296** | | **74,768 float ops** |

**Costo de simulación: ~74,768 float ops para simular 2,296 ops SHA-256 (factor 32.6x)**

### Diferencias fundamentales

1. **Tipo de dato**: SHA-256 opera en enteros unsigned 32-bit. Transformer opera en float32. Son dominios distintos.
2. **Reversibilidad**: XOR es reversible (a⊕b⊕b = a). Las multiplicaciones float del transformer no son reversibles (pérdida de precisión).
3. **Determinismo**: SHA-256 es 100% determinista. Transformer tiene no-determinismo por floating point accumulation order.
4. **Paralelismo**: Las 64 rondas de SHA-256 son secuenciales (cada ronda depende de la anterior). Las matmuls del transformer son altamente paralelizables.
5. **Avalancha**: SHA-256 tiene efecto avalancha (1 bit cambia → ~50% bits cambian). El transformer tiene efecto similar vía softmax + residual connections pero sobre distribuciones continuas, no bits.
