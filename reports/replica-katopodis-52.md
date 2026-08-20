# Replicação de §5.2.2 — circular contra termômetro simples

A afirmação sob teste, do capítulo 5 da dissertação:

> *"For both tasks, the circular thermometer reaped better results than the
> simple one, with the difference being more pronounced with lunar lander."*
> — §5.2.2

Reprodução:

```bash
uv run python scripts/replica_katopodis_52.py --episodios 150 --sementes 20   # config do artigo
uv run python scripts/replica_katopodis_52.py --episodios 150 --sementes 20 \
    --resolucao 32 --tupla 16 --passos 200 --exploracao 0.01                  # baixa resolução
```

---

## 1. Resultado: a direção depende do regime, e no regime do artigo dá empate

CartPole-v1, Algoritmo 4.1 (SARSA de `m` passos com esquecimento), `φ = 0,85`,
`γ = 1,0`. Retorno médio do último quinto dos episódios, 20 sementes pareadas,
IC95% por bootstrap pareado.

| configuração | simples | circular | razão | dif. pareada, IC95% | veredito |
|---|---|---|---|---|---|
| **do artigo** — `R=128, n=64, m=500, ε=0,005` | 176,7 ± 32,0 | 169,2 ± 53,7 | **0,96×** | −7,5 [−32,8; +16,9] | **empate** |
| baixa resolução — `R=32, n=16, m=200, ε=0,01` | 69,7 ± 16,2 | **117,5 ± 13,9** | **1,68×** | **+47,8 [+39,6; +56,1]** | **circular vence** |

Na baixa resolução o circular vence em **20 de 20 sementes**, e a diferença
**cresce** com mais episódios (medido antes: 65 contra 139 em 400 episódios), o
que descarta falta de orçamento. Na configuração do artigo o IC cruza zero com
folga.

## 2. Interpretação

O mecanismo do ganho do circular é a **similaridade radial**: valores extremos
do intervalo viram padrões vizinhos. Medido em
`tests/test_identidades_artigo.py`, com `R = 16`:

| distância de Hamming | circular | termômetro |
|---|---|---|
| d(0,00 · 0,99) | **2** | 16 |
| d(0,00 · 0,50) | 16 | 8 |

Isso importa **quando a resolução é escassa**. Com `R = 128` o termômetro já
tem discriminação de sobra e o ganho do wrap-around se dilui. É consistente com
a própria ressalva do texto — a diferença é "more pronounced with lunar lander",
isto é, **varia com o problema**; o que este experimento acrescenta é que varia
também com a resolução.

Nota: o circular gasta o dobro de resolução para a mesma discriminação, porque o
número de bits ativos é fixo em `⌊R/2⌋`. Comparar `R = 32` circular com `R = 32`
termômetro não é comparar orçamentos iguais de informação — é o desenho do
artigo, e vale registrar a assimetria.

## 3. Quatro ressalvas, e nenhuma é pequena

1. **Orçamento de ~1/50 do original.** O artigo usa 50 réplicas e 1 M passos;
   aqui são 20 sementes e 150 episódios (≈ 25 k passos na config do artigo). As
   curvas dele podem separar bem depois de onde paramos. **Isto não refuta
   §5.2.2** — mostra que a afirmação não se sustenta *no orçamento pequeno* e
   *na configuração declarada*.
2. **O intervalo `[l, u] = [−1,5; 1,5]` é palpite nosso.** A dissertação não o
   dá para CartPole no texto que li; o valor vem das configurações de tune do
   repositório em PyTorch. A codificação depende dele diretamente.
3. **Recortamos a observação em `[l, u]`; o artigo não.** A eq. (3.24) pressupõe
   `x` no intervalo de definição, e fora dele o circular **dá a volta**,
   apelidando extremos em valores centrais. Sem recorte mediríamos essa
   patologia. O termômetro satura em vez de dar a volta, então o recorte o afeta
   menos — **assimetria a favor do termômetro**, não contra.
4. **Variância alta.** Na config do artigo, a semente 12 do circular colapsou em
   15 de retorno, contra 311 na semente 13. Com 20 sementes isso ainda mexe na
   média.

## 4. O que fica para o projeto

**O codificador de produção não muda.** Segue o `TermometroQuantil` da SPEC-01,
por três razões independentes deste experimento:

- Ele venceu a medição direta no **nosso** problema (4,3° contra 7,6°).
- O circular **não tem padrão todo-zero** — acende sempre `⌊R/2⌋` bits —, então
  quebra a convenção que dispensa o bit de validade. As tarefas do artigo não
  têm slot vazio; a nossa tem.
- Este resultado não dá vantagem ao circular no regime de resolução alta.

**O que vale reter:** se a SPEC-01 for revisitada com resolução baixa — e há
motivo, porque `T = 8` já é baixo —, o circular passa a ser candidato sério
para as dimensões **angulares**, onde a similaridade radial é semanticamente
correta e não só conveniente. Isso conecta com a **C1**: o rumo `θ ∈ [0, 2π)` é
exatamente o caso em que o wrap-around é a topologia certa. O circular não fecha
C1 (que é sobre a *ação*), mas fecharia a parte de *entrada* se o rumo entrar na
observação.
