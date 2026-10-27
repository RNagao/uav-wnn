# Pico de RAM e latência — 2d/24s, 30 medições por modelo

Gerado por `scripts/spec03_latencia_memoria.py --repeticoes 30` em 2026-10-26. Cada medição é um **processo novo** (o `VmHWM` é monotônico; repetir no mesmo processo não mede nada), com a mesma observação pré-calculada no pai e o protocolo de latência da SPEC-03 (menor mediana de 8×1000). Média ± desvio-padrão amostral; entre parênteses, mínimo–máximo.

## pico total (MB)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **46.17 ± 0.06** | 46.06–46.34 | 30 |
| wnn-T32 (NumPy) | **53.56 ± 0.09** | 53.37–53.77 | 30 |
| maddpg (PyTorch) | **506.14 ± 0.06** | 506.05–506.30 | 30 |

## arcabouço (MB)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **31.50 ± 0.05** | 31.38–31.57 | 30 |
| wnn-T32 (NumPy) | **31.51 ± 0.05** | 31.38–31.58 | 30 |
| maddpg (PyTorch) | **496.14 ± 0.06** | 496.06–496.30 | 30 |

## acréscimo do modelo (MB)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **14.67 ± 0.03** | 14.64–14.79 | 30 |
| wnn-T32 (NumPy) | **22.06 ± 0.09** | 21.91–22.25 | 30 |
| maddpg (PyTorch) | **10.00 ± 0.01** | 9.98–10.01 | 30 |

## RSS pós-carga (MB)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **40.55 ± 0.20** | 40.39–41.57 | 30 |
| wnn-T32 (NumPy) | **41.07 ± 0.05** | 40.91–41.16 | 30 |
| maddpg (PyTorch) | **499.38 ± 0.06** | 499.30–499.55 | 30 |

## latência mediana (µs)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **22.54 ± 0.38** | 21.87–23.58 | 30 |
| wnn-T32 (NumPy) | **24.96 ± 0.54** | 24.19–27.29 | 30 |
| maddpg (PyTorch) | **48.47 ± 0.55** | 47.30–50.13 | 30 |

## latência p99 (µs)

| modelo | média ± desvio | min–max | n |
| --- | --- | --- | --- |
| wnn-T16 (NumPy) | **67.03 ± 12.27** | 51.38–104.79 | 30 |
| wnn-T32 (NumPy) | **70.60 ± 9.43** | 58.33–110.96 | 30 |
| maddpg (PyTorch) | **145.52 ± 11.87** | 126.76–186.21 | 30 |

![box plot das 30 medições](spec03-pico-ram-rep30-a2s24-ref.png)
