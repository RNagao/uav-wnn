# Pico de RAM e latência — 8d/24s, 30 medições por modelo

Gerado por `scripts/spec03_latencia_memoria.py --repeticoes 30` em 2026-10-26. Cada medição é um **processo novo** (o `VmHWM` é monotônico; repetir no mesmo processo não mede nada), com a mesma observação pré-calculada no pai e o protocolo de latência da SPEC-03 (menor mediana de 8×1000). Média ± desvio-padrão amostral; entre parênteses, mínimo–máximo.

## pico total (MB)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **48.01 ± 0.16** | 47.78–48.32 | 30 |
| wnn-T32 (NumPy) | **59.45 ± 0.16** | 59.13–59.71 | 30 |
| maddpg (PyTorch) | **506.12 ± 0.06** | 506.02–506.28 | 30 |

## arcabouço (MB)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **31.49 ± 0.04** | 31.42–31.59 | 30 |
| wnn-T32 (NumPy) | **31.49 ± 0.04** | 31.39–31.58 | 30 |
| maddpg (PyTorch) | **496.11 ± 0.06** | 496.02–496.27 | 30 |

## acréscimo do modelo (MB)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **16.51 ± 0.16** | 16.30–16.82 | 30 |
| wnn-T32 (NumPy) | **27.96 ± 0.15** | 27.63–28.14 | 30 |
| maddpg (PyTorch) | **10.01 ± 0.01** | 9.99–10.03 | 30 |

## RSS pós-carga (MB)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **40.88 ± 0.03** | 40.82–40.97 | 30 |
| wnn-T32 (NumPy) | **41.63 ± 0.26** | 40.67–41.79 | 30 |
| maddpg (PyTorch) | **499.36 ± 0.06** | 499.27–499.53 | 30 |

## latência mediana (µs)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **23.96 ± 0.27** | 23.44–24.40 | 30 |
| wnn-T32 (NumPy) | **27.30 ± 0.61** | 26.64–30.06 | 30 |
| maddpg (PyTorch) | **48.56 ± 0.73** | 47.62–50.72 | 30 |

## latência p99 (µs)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **68.15 ± 12.20** | 55.63–107.16 | 30 |
| wnn-T32 (NumPy) | **78.11 ± 12.46** | 60.28–112.99 | 30 |
| maddpg (PyTorch) | **151.90 ± 17.48** | 117.16–198.07 | 30 |

![box plot das 30 medições](spec03-pico-ram-rep30-a8s24-v2.png)
