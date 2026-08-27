# T = 16 × T = 32 sob ruído de GPS — escada L→R, 2 sementes

O `T = 16` (econômico, 34 RAMs, 1,2 MB) foi levado ao cenário ruidoso com o mesmo
protocolo do `T = 32` (desempenho, 67 RAMs, 2,6 MB): escada de ablação 0–4,
σ ∈ {1, 3, 5, 10} m, 500 episódios pareados por CRN, regime L→R (treino limpo,
avaliação ruidosa).

## 0. Antes de tudo: o baseline publicado é reproduzível aqui

O `melhor` do `T = 32` usado na SPEC-08 foi treinado em outra máquina. Reproduzi o
degrau 0 com o checkpoint local, **nos mesmos 500 episódios**:

    publicado (outra máquina): ct_medio 130.412 · SR 0.808
    local     (esta máquina) : ct_medio 130.412 · SR 0.808

Idênticos. A avaliação é determinística (`explore=False`, CRN por semente), logo
duas redes diferentes divergiriam em ao menos um dos 500 episódios. É equivalência
**funcional** comprovada — não comparação bit a bit dos `.npz`, que exigiria o
arquivo da outra máquina.

## 1. Qual checkpoint — e por que isso mudou o resultado

A SPEC-08 rodou sobre o `checkpoints/melhor`. Como registrado na §9 do
`spec06-grid-search.md`, esse checkpoint congelou cedo (2,25–2,5 M) porque o banco
de validação satura. A diferença é grande:

| modelo (semente 0) | `melhor` | `final` |
|---|---|---|
| T = 32 | 130,4 s · SR 80,8 % | **87,5 s · SR 89,6 %** |
| T = 16 | 73,1 s · SR 92,8 % | **52,4 s · SR 96,8 %** |

Comparar os dois pelo `melhor` daria ao `T = 16` uma vantagem de 57 s e 12 pp que é
**artefato de seleção de checkpoint**, não do modelo. Tudo abaixo usa o `final` —
a melhor rede que cada corrida produziu.

## 2. O resultado principal: os dois são indistinguíveis

Teto limpo (degrau 0), redes finais, 500 episódios:

| | semente 0 | semente 1 | média |
|---|---|---|---|
| T = 16 | 52,4 s · 96,8 % | 106,0 s · 86,0 % | **79,2 s · 91,4 %** |
| T = 32 | 87,5 s · 89,6 % | 57,2 s · 96,0 % | **72,4 s · 92,8 %** |

Sob ruído, média das duas sementes (CT s / SR):

| σ | degrau | T = 16 | T = 32 | Δ SR |
|---|---|---|---|---|
| 1 | 1 bruto | 63,2 / 94,9 % | 60,8 / 95,4 % | −0,5 pp |
| 1 | 4 KF+ctrl | 66,0 / 94,3 % | 60,7 / 95,4 % | −1,1 pp |
| 3 | 1 bruto | 58,2 / 96,2 % | 53,5 / 97,1 % | −0,9 pp |
| 3 | 4 KF+ctrl | 58,6 / 96,0 % | 54,3 / 96,8 % | −0,8 pp |
| 5 | 1 bruto | 52,3 / 97,7 % | 52,7 / 97,5 % | **+0,2 pp** |
| 5 | 4 KF+ctrl | 55,2 / 96,8 % | 57,9 / 96,0 % | **+0,8 pp** |
| 10 | 1 bruto | 55,1 / 98,0 % | 56,6 / 97,7 % | **+0,3 pp** |
| 10 | 4 KF+ctrl | 56,0 / 96,8 % | 56,8 / 96,4 % | **+0,4 pp** |

**Δ SR fica em ±2,6 pp nas 16 células, e abaixo de 1 pp em 12 delas.** O `T = 16`
entrega o mesmo desempenho sob ruído com **metade da memória**.

**A semente domina o modelo.** O spread entre sementes (10,8 pp no T = 16; 6,4 pp
no T = 32) é maior que qualquer diferença entre os dois. Com **uma** semente eu
teria concluído que o `T = 16` ganha por 7 pp — na semente 0 ele é o bom e o
`T = 32` é o ruim; na semente 1 inverte. É a mesma armadilha da §9.

## 3. O KF: 1 vitória em 24 células

Contando todas as células rodadas (2 modelos × 2 sementes × 4 σ, mais as duas do
`melhor`), o degrau 4 supera algum degrau mais barato em **uma**:

    T = 32 semente 1, σ = 10:  degrau 4 vs degrau 1
    Δ = −11,3 s   IC[−19,0; −4,4]   (bootstrap pareado, 500 ep, Holm)

É o **primeiro resultado positivo do KF no projeto**, e ele aparece no **maior σ** —
coerente com a leitura da SPEC-08 §6: o filtro morde quando o ruído chega a fazer
dano, e σ ≤ 5 é brando demais para isso. Uma célula em 24 não sustenta veredito;
sustenta a próxima varredura, em σ maior.

## 4. O dithering vale para os dois modelos

Fração dos travamentos do teto limpo que o **ruído bruto** destrava:

| modelo | trava em | σ=1 | σ=3 | σ=5 | σ=10 |
|---|---|---|---|---|---|
| T=16 s0 | 3,2 % | 62 % | 75 % | 88 % | 69 % |
| T=16 s1 | 14,0 % | 36 % | 51 % | 70 % | 79 % |
| T=32 s0 | 10,4 % | 33 % | 52 % | 62 % | 79 % |
| T=32 s1 | 4,0 % | 45 % | 80 % | 75 % | 40 % |

O achado da SPEC-08 **não era do `T = 32`**: os quatro casos ganham com ruído, e
recuperam frações parecidas (51–88 % em σ ≥ 3). A taxa de travamento varia com a
semente, não com o modelo — o ganho absoluto varia junto, mas o mecanismo é o mesmo.

## 5. O atestado RF-05 é por modelo — e a grade de `q_w` era curta

O `T = 16` semente 1 **reprovou** no gate com a grade padrão `q_w ∈ {0; 0,25; 1; 4}`:
NEES 2,80 (otimista) no maior valor disponível. Passou com `q_w = 16`
(NEES 1,63, NIS consistente). É o mesmo fenômeno já registrado na SPEC-08 §2 —
mobilidade mais agressiva exige mais `Q` — só que a grade não ia longe o bastante.

`Q_W_GRADE` continua curta por padrão, para não mudar o `q_w` de atestados já
emitidos; `--q-w-grade` estende quando preciso. Um caminho fixo de gate faria a
escada de um modelo rodar com o `q_w` de outro, por isso `--gate` também virou
parâmetro em `spec08_escada.py`.

Atestados: `q_a = 200` em todos; `q_w = 4,0` em T16-s0, T32-s0, T32-s1; **`q_w = 16`
em T16-s1**.

## 6. O que fica

- **O `T = 16` é viável sob ruído.** Mesmo desempenho, metade da memória. Para a
  fronteira custo×desempenho isso é o resultado que interessa.
- **Duas sementes ainda são poucas.** O spread entre sementes é maior que o efeito
  medido; o veredito pleno pede ≥ 5 (T-08.2).
- **G2 segue não declarado.** O critério 3 (degrau 4 > degraus 2 e 3) falha em 24 de
  24 células — a única vitória do KF é sobre o degrau 1.
- **Próximo passo natural:** σ > 10 m, onde a única vitória do KF apareceu.

## 7. Reprodução

```bash
# atestado por modelo (estenda a grade se reprovar por NEES otimista)
uv run python scripts/spec08_gate_rf05.py --checkpoint <run>/checkpoints/final \
  --aproximador ntupla_dissertacao --episodios 100 \
  [--q-w-grade 0,0.25,1,4,16,64] --saida results/etapa2/gate_rf05-<rot>.json

# escada, apontando para o atestado daquele modelo
uv run python scripts/spec08_escada.py --checkpoint <run>/checkpoints/final \
  --aproximador ntupla_dissertacao --episodios 500 \
  --gate results/etapa2/gate_rf05-<rot>.json \
  --saida results/etapa2/escada_lr-<rot>.json
```
