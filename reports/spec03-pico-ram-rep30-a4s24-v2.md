# Pico de RAM e latência — 4d/24s, 30 medições por modelo

Gerado por `scripts/spec03_latencia_memoria.py --repeticoes 30` em 2026-10-26. Cada medição é um **processo novo** (o `VmHWM` é monotônico; repetir no mesmo processo não mede nada), com a mesma observação pré-calculada no pai e o protocolo de latência da SPEC-03 (menor mediana de 8×1000). Média ± desvio-padrão amostral; entre parênteses, mínimo–máximo.

## pico total (MB)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **46.62 ± 0.08** | 46.45–46.82 | 30 |
| wnn-T32 (NumPy) | **79.37 ± 0.08** | 79.20–79.50 | 30 |
| maddpg (PyTorch) | **506.12 ± 0.06** | 506.02–506.25 | 30 |

## arcabouço (MB)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **31.49 ± 0.05** | 31.41–31.61 | 30 |
| wnn-T32 (NumPy) | **31.49 ± 0.05** | 31.39–31.62 | 30 |
| maddpg (PyTorch) | **496.13 ± 0.06** | 496.03–496.26 | 30 |

## acréscimo do modelo (MB)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **15.13 ± 0.06** | 14.99–15.25 | 30 |
| wnn-T32 (NumPy) | **47.88 ± 0.06** | 47.74–47.96 | 30 |
| maddpg (PyTorch) | **9.99 ± 0.02** | 9.96–10.05 | 30 |

## RSS pós-carga (MB)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **40.62 ± 0.05** | 40.53–40.73 | 30 |
| wnn-T32 (NumPy) | **42.72 ± 0.14** | 42.51–42.98 | 30 |
| maddpg (PyTorch) | **499.37 ± 0.05** | 499.27–499.50 | 30 |

## latência mediana (µs)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **23.23 ± 0.34** | 22.65–24.23 | 30 |
| wnn-T32 (NumPy) | **25.83 ± 0.33** | 25.18–26.85 | 30 |
| maddpg (PyTorch) | **48.61 ± 1.55** | 46.96–55.69 | 30 |

## latência p99 (µs)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **66.19 ± 9.51** | 51.81–90.59 | 30 |
| wnn-T32 (NumPy) | **74.11 ± 10.85** | 54.69–99.79 | 30 |
| maddpg (PyTorch) | **150.11 ± 17.50** | 121.49–205.72 | 30 |

![box plot das 30 medições](spec03-pico-ram-rep30-a4s24-v2.png)
