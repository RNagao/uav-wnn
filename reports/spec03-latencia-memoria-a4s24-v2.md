# Latência e pico de RAM — 4d/24s, medidos

Gerado por `scripts/spec03_latencia_memoria.py` em 2026-10-26 · x86_64 · mesma observação para os três · um processo por modelo · menor mediana de 8×1000.

## Tempo de inferência (decisão completa: obs → ação)

| modelo | mediana | p99 |
| --- | --- | --- |
| wnn-T16 | 23.4 µs | 64.2 µs |
| wnn-T32 | 25.4 µs | 63.2 µs |
| maddpg | 48.0 µs | 161.8 µs |

- wnn-T16 é **2.1×** mais rápida que o MADDPG

- wnn-T32 é **1.9×** mais rápida que o MADDPG

## Pico de RAM do processo (VmHWM)

| modelo | arcabouço | pico total | acréscimo do modelo | RSS pós-carga |
| --- | --- | --- | --- | --- |
| wnn-T16 (NumPy) | 31.5 MB | **46.5 MB** | 15.1 MB | 40.6 MB |
| wnn-T32 (NumPy) | 31.5 MB | **79.5 MB** | 47.9 MB | 42.6 MB |
| maddpg (PyTorch) | 496.2 MB | **506.2 MB** | 10.0 MB | 499.4 MB |

- **pico total**: a marca d'água alta de RSS do processo ao fim do laço de inferência — o que o dispositivo precisa ter de RAM para carregar o modelo e decidir. É a figura de memória do projeto;
- **arcabouço**: o pico depois dos imports e antes de qualquer peso — NumPy na WNN, NumPy + PyTorch no MADDPG. É custo real de quem implanta com aquele arcabouço, mas é biblioteca, não rede;
- **acréscimo do modelo**: `pico total − arcabouço` — quanto carregar a rede (com os transientes do load: na WNN, a rede de dicionários que vira LUT plana) e executá-la eleva o pico;
- **RSS pós-carga**: o residente estável com o modelo carregado, após `malloc_trim` — o que FICA, não o pico;
- o simulador fica fora: a observação é calculada no processo pai e lida pelos filhos de um `.npy`.

![tempos](spec03-latencia-a4s24.png)
![pico de RAM](spec03-pico-ram-a4s24.png)
