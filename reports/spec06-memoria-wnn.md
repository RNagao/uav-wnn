# Memória da WNN — onde ela está, o que é do modelo e o que dá para cortar

Pergunta de partida: *"a rede sem peso não deveria ter mais parâmetros que a
rede neural convencional"*. Resposta curta: **a implementação não desperdiça —
a ocupação é do modelo e da carga de trabalho; e com poda pós-treino a WNN fica
MENOR que o ator do MADDPG, com custo de SR medido e pequeno.**

## 1. A comparação com o wisardpkg

Fonte lida: `IAZero/wisardpkg`, `src/models/wisard/ram.cc`. A RAM deles é

```cpp
std::unordered_map<addr_t, content_t> positions;   // esparso, por RAM
// train():  positions[getIndex(image)]++          // 1 contador int por entrada
// getsizeof(): positions.size() × (sizeof(addr_t) + sizeof(content_t))
```

A nossa (`redes.py`) é `dict[int, ndarray]` por RAM — **a mesma estrutura
esparsa**, e a contabilidade `ordenado` do nosso T1 é exatamente a fórmula do
`getsizeof` deles. Não há desperdício estrutural em relação à referência.

As diferenças são de *conteúdo* e de *carga*:

| | wisardpkg (classificação) | nossa (ator, Alg. 4.4) |
|---|---|---|
| carga por entrada | 1 contador `int` | **3 × float** (u₁, u₂, μ_s) |
| treino | UMA passada num dataset finito | milhões de passos de RL |
| ocupação limitada por | nº de amostras de treino | **diversidade da experiência** |

A intuição "WNN é pequena" vem da WiSARD de classificação: dataset finito ⇒
ocupação limitada. O nosso ator é outra coisa — a eq. (4.16) mostra que ele é
uma **máquina de núcleo**: `f_t(x) = Σ_j α_j Δ_j k(x, x_j)`, uma soma sobre
TODAS as atualizações. A memória cresce com o suporte da distribuição de
estados visitada. Depois de 6,75 M de passos: 70 950 entradas (26 % das
tabelas), 975 KB na variante `ordenado`, contra 287 KB do ator MADDPG.

## 2. Mas o suporte está quase todo morto

Concentração de massa no ator T32 (1/12): **50 % da massa em 6,3 % das
entradas; 90 % em 41 %**. A cauda são contribuições minúsculas de estados
visitados poucas vezes na exploração — e na convenção de média cada entrada
contribui `v/N` à saída, então |v| pequeno quase não move a ação.

**Poda pós-treino** (remover as entradas de menor |v|), medida com 500
sementes do banco de teste:

```
T32 a1s12 (SR base 80,8%)          T16 a1s12 (SR base 96,8%)
poda  entradas    KB     SR         poda  entradas    KB     SR
  0%   70 950   970K  80,8%          0%    36 547   500K  96,8%
 70%   21 285   291K  81,2%         70%    10 965   150K  96,0%
 90%    7 095    97K  81,4%         90%     3 655    50K  93,8%
 98%    1 419    20K  82,0%         98%       731    10K  90,0%
```

- Na política fraca (T32 `melhor`, congelado cedo) a poda é **de graça até
  98 %** — a tabela é quase toda ruído.
- Na política forte (T16) a curva é real: **70 % custa 0,8 pp** (dentro do
  ruído de ±0,9 pp); 90 % custa 3 pp; 98 % custa 7 pp e 32 s de CT.

## 3. A cadeia completa, medida

| artefato | T16 a1s12 | vs MADDPG (287 KB) |
|---|---|---|
| checkpoint de treino (ator+crítico, f64/u64) | 1 333 KB | 4,6× |
| deploy (só ator, f32/u16) | 472 KB | 1,6× |
| **deploy podado 70 %** | **144 KB** | **0,50× — MENOR** |
| deploy podado 90 % (SR −3 pp) | ~50 KB | 0,17× |

Export e verificação: `spec06_exportar_deploy.py --poda 0.7`. Um export podado
muda a ação por construção, então a verificação é por **comportamento** (SR no
banco de teste, queda tolerada ≤ 2 pp com 200 sementes), não por igualdade de
ação. Medido: 97,5 % → 96,0 %, aprovado.

## 4. O que fica em aberto

- **`n = 8` nunca foi testado** — as grades só cobriram `n ∈ {12, 24}` (e
  `n = 24` foi muito pior: cob média 5,7 % contra 19,1 %). Com `n = 8`,
  `N = 100` RAMs de 2⁸ = 256 posições cabem DENSAS em 300 KB, sem chave nenhuma
  e com indexação O(1) — memória limitada na inicialização, como uma matriz de
  pesos. É a mudança estrutural que tornaria a WNN pequena *por construção*,
  mas exige retreinar e re-grade.
- A poda foi verificada em `Na = 1`, regime limpo. Antes de trocar os artefatos
  das campanhas, verificar por célula e sob ruído.
- A poda usa |v|∞ por entrada; critérios por contribuição efetiva (frequência
  de acesso na avaliação) não foram testados — o ator não guarda contadores
  (§4.3.1: "they hold only values").

## 5. Reprodução

```bash
uv run python scripts/spec06_exportar_deploy.py --todos              # f32/u16
uv run python scripts/spec06_exportar_deploy.py <dir> --na 1 --ns 12 --poda 0.7
```
