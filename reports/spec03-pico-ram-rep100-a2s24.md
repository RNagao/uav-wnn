# Pico de RAM e latência — 2d/24s, 100 medições por modelo

Gerado por `scripts/spec03_latencia_memoria.py --repeticoes 100` em 2026-09-10. Cada medição é um **processo novo** (o `VmHWM` é monotônico; repetir no mesmo processo não mede nada), com a mesma observação pré-calculada no pai e o protocolo de latência da SPEC-03 (menor mediana de 8×1000). Média ± desvio-padrão amostral; entre parênteses, mínimo–máximo.

## pico total (MB)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **45.93 ± 0.11** | 45.57–46.14 | 100 |
| wnn-T32 (NumPy) | **53.26 ± 0.15** | 52.97–53.60 | 100 |
| maddpg (PyTorch) | **505.98 ± 0.06** | 505.84–506.16 | 100 |

## arcabouço (MB)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **31.28 ± 0.05** | 31.16–31.43 | 100 |
| wnn-T32 (NumPy) | **31.29 ± 0.05** | 31.14–31.41 | 100 |
| maddpg (PyTorch) | **495.96 ± 0.06** | 495.83–496.16 | 100 |

## acréscimo do modelo (MB)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **14.65 ± 0.10** | 14.34–14.80 | 100 |
| wnn-T32 (NumPy) | **21.97 ± 0.14** | 21.70–22.27 | 100 |
| maddpg (PyTorch) | **10.02 ± 0.01** | 9.99–10.06 | 100 |

## RSS pós-carga (MB)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **40.58 ± 0.45** | 40.10–41.37 | 100 |
| wnn-T32 (NumPy) | **40.84 ± 0.08** | 40.61–41.01 | 100 |
| maddpg (PyTorch) | **499.18 ± 0.06** | 499.04–499.37 | 100 |

## latência mediana (µs)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **22.58 ± 0.34** | 22.00–24.22 | 100 |
| wnn-T32 (NumPy) | **25.17 ± 0.47** | 24.28–27.40 | 100 |
| maddpg (PyTorch) | **49.08 ± 0.83** | 47.82–52.79 | 100 |

## latência p99 (µs)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **74.92 ± 14.72** | 52.04–158.77 | 100 |
| wnn-T32 (NumPy) | **82.67 ± 14.81** | 59.05–136.57 | 100 |
| maddpg (PyTorch) | **167.37 ± 27.15** | 128.18–334.70 | 100 |
