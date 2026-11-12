# Pico de RAM e latência — 2d/24s, 30 medições por modelo

Gerado por `scripts/spec03_latencia_memoria.py --repeticoes 30` em 2026-10-26. Cada medição é um **processo novo** (o `VmHWM` é monotônico; repetir no mesmo processo não mede nada), com a mesma observação pré-calculada no pai e o protocolo de latência da SPEC-03 (menor mediana de 8×1000). Média ± desvio-padrão amostral; entre parênteses, mínimo–máximo.

## pico total (MB)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **45.88 ± 0.09** | 45.70–46.04 | 30 |
| wnn-T32 (NumPy) | **73.38 ± 0.10** | 73.11–73.56 | 30 |
| maddpg (PyTorch) | **506.11 ± 0.05** | 506.02–506.25 | 30 |

## arcabouço (MB)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **31.49 ± 0.04** | 31.37–31.57 | 30 |
| wnn-T32 (NumPy) | **31.49 ± 0.06** | 31.35–31.62 | 30 |
| maddpg (PyTorch) | **496.12 ± 0.05** | 496.01–496.26 | 30 |

## acréscimo do modelo (MB)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **14.39 ± 0.07** | 14.22–14.52 | 30 |
| wnn-T32 (NumPy) | **41.89 ± 0.08** | 41.69–42.06 | 30 |
| maddpg (PyTorch) | **9.99 ± 0.02** | 9.98–10.05 | 30 |

## RSS pós-carga (MB)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **41.28 ± 0.05** | 41.16–41.37 | 30 |
| wnn-T32 (NumPy) | **42.20 ± 0.42** | 41.59–42.62 | 30 |
| maddpg (PyTorch) | **499.35 ± 0.06** | 499.24–499.50 | 30 |

## latência mediana (µs)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **22.40 ± 0.30** | 21.62–22.96 | 30 |
| wnn-T32 (NumPy) | **25.19 ± 0.48** | 24.57–26.90 | 30 |
| maddpg (PyTorch) | **48.82 ± 0.97** | 47.52–51.98 | 30 |

## latência p99 (µs)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **62.76 ± 7.50** | 52.43–81.68 | 30 |
| wnn-T32 (NumPy) | **71.37 ± 10.63** | 57.96–100.77 | 30 |
| maddpg (PyTorch) | **151.85 ± 26.35** | 125.00–240.42 | 30 |

![box plot das 30 medições](spec03-pico-ram-rep30-a2s24-v2.png)
