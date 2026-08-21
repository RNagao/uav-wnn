# T-05.4(b) — a ação circular: normal projetada contra gaussiana simples

**Portão da SPEC-05, critério de saída 2. Veredito: ABERTO.** A decisão da §4.3 se
sustenta, com IC separado por 12× o MDE.

O critério foi escrito para falsificar a nossa própria decisão:

> C2 (normal projetada) estritamente superior a C0 (gaussiana simples) no
> brinquedo, com IC separado. Se C2 **não** vencer, a §4.3 estava errada e a
> decisão precisa ser revista antes da SPEC-06 — não depois.

## Resultado

Bandido contextual, 400 alvos de avaliação pareados, 3 sementes de treino, melhor
configuração de cada braço após varredura própria de `σ` **e** de taxa:

| | erro médio | mediana | p90 | < 15° |
|---|---|---|---|---|
| C0 (centrado, `σ`=0,10, `α`=1e−3) | 16,6° | 10,9° | 37,0° | 64 % |
| **C2 (`σ`=0,20, `α`=1e−3)** | **3,2°** | **2,2°** | **7,1°** | **99 %** |

```
C2 − C0   Δ = −13,45°   IC95 % [−14,53; −12,43]   MDE 1,06°
          vitória do C2 em 98,5 % dos alvos [97,2 %; 99,8 %]
          Wilcoxon p = 4,4·10⁻⁶⁷      n = 400 pareado
```

O IC está a 12 MDEs de zero. Não é resultado marginal.

## O caminho até aqui, porque ele contém dois erros meus

### Erro 1 — o primeiro portão media a coisa errada

A primeira versão usou apenas o brinquedo sequencial `toy/nav.py` e deu
**inconclusivo, com os dois braços piores que rumo aleatório**: 150° de erro
mediano, 3–8 % de sucesso, quando um rumo sorteado erraria ~90° em média.

Diagnóstico: `δ ≈ −19` **constante** até o fim do treino. Com sucesso quase nulo o
retorno vale ≈ −60 qualquer que seja o rumo, então o gradiente não carregava
informação direcional alguma. Aquilo mediria **exploração sob recompensa esparsa**,
que é pergunta diferente da que o portão faz.

Um braço que não aprende não é evidência sobre o outro. Um empate entre dois
fracassos não é empate — é ausência de medição, e reportá-lo como empate teria sido
pior que não medir.

### Erro 2 — `σ` significa coisas diferentes nos dois esquemas

Varrer só a taxa parecia justo e não era. No C0, `σ` é desvio em `θ_norm`: medido,
`σ = 0,25` são **90° de ruído angular**, com que nenhuma política aponta para nada.
No C2, `σ` é desvio no vetor de R³, e o ruído angular resultante depende de
`‖f₁:₂‖` — a política pode **encolher o próprio ruído** crescendo a norma.

Essa assimetria é a vantagem sob teste. Fixar um `σ` comum teria construído a
conclusão em vez de medi-la.

## Decomposição em três instrumentos

Corrigidos os dois erros, o experimento virou três, do mais isolado ao mais
completo:

### (A) Bandido fixo — isola o mecanismo

Um estado, um alvo, `r = cos(θ − θ*)`. Sem crédito temporal, sem generalização.

| | `r` média | `r` pior | > 0,95 |
|---|---|---|---|
| C0 cru | 0,404 | −0,866 | 8 % |
| C0 centrado | 0,496 | −0,500 | 8 % |
| **C2** | **0,999** | **0,995** | **100 %** |

**A coluna do C0 é exatamente `cos(alvo)`.** Ele aponta sempre para 0° e recebe o
que a sorte do alvo der. Os 8 % que acerta são precisamente as 3 sementes do alvo
em 0°, onde a fronteira do recorte já aponta por acidente — e isso vale para os
três valores de `σ`, com e sem centralização.

O mecanismo é exato e vale escrever: com `f₁` abaixo de zero com margem, **toda**
amostra recorta para a mesma ação. A recompensa deixa de variar, `(r − b) → 0`, e o
gradiente morre. É **zona absorvente** — não é lentidão de convergência, é ausência
de sinal, e nenhuma quantidade de episódios sai de lá.

### (B) Bandido contextual — o portão

Alvo sorteado por tentativa, observação com a **mesma normalização do ambiente de
referência** (`(p − s + S)/(2S)`). Exige generalizar entre estados; sem crédito
temporal. É o teste mais forte em que **os dois braços demonstravelmente
aprendem** — e é por isso que só ele decide.

Nota sobre magnitude: aqui o C0 vai muito melhor que em (A) — 16,6° contra o
desastre do bandido fixo. A generalização entre estados o tira da zona absorvente,
porque atualizações vindas de alvos distantes da fronteira arrastam os pesos
compartilhados. A vantagem do C2 permanece, mas é **5×**, não infinita, e é este o
número honesto.

### (C) Nav sequencial — reportado, e **fora do veredito**

| | passos | erro | sucesso |
|---|---|---|---|
| C0 cru (`σ`=0,10) | 40,0 | 89,4° | 0,0 % |
| C0 centrado (`σ`=0,10) | 37,6 | 133,0° | 8,5 % |
| C2 (`σ`=0,10) | 39,5 | 98,4° | 2,5 % |

A política ótima analítica faz **7,86 passos com 100 % de sucesso**, então o
brinquedo é resolvível e a falha é nossa. Ela persiste com `σ` fixa, então não é o
anelamento; e persiste com recompensa densa no progresso, então não é só
exploração. Resta o crédito temporal, e ele falha **nos dois braços**.

Isto fica como **pendência aberta**, não como resultado. Comparar dois fracassos
não informa sobre a §4.3, e incluí-los no veredito só diluiria (B) com ruído.

## Contra espantalho — quatro concessões ao C0

1. **Varredura de `σ` própria** (3 valores), pela razão da seção anterior.
2. **Varredura de taxa própria** (2 valores). A SPEC-03 já pagou a lição de que a
   taxa é não monotônica com ótimo agudo, e aqui ela reapareceu: no primeiro
   ensaio, com taxa única de 3e−3, o C2 media 19,4° e o C0 27,0°; varrendo, os dois
   melhoram e o C2 vai a 3,2°. A conclusão não mudou de sinal, mas a magnitude
   mudou por 6×, e reportar a versão mal sintonizada teria sido reportar a nossa
   sintonia, não os esquemas.
3. **Variante centrada em 0,5** — o centro da caixa de ação, que é o que qualquer
   implementação competente faria. Ela foi de fato a melhor do C0 em (A) e (B), e é
   a que entra na comparação. Sem isto valeria a objeção "você inicializou mal o
   baseline".
4. **`d = 2` contra `d = 3`** — o C0 tem uma cabeça de menos por posição de
   memória, então a comparação o favorece em custo.

Nenhuma das quatro salvou o C0, o que é o que torna o resultado utilizável.

## O que isto fecha e o que não fecha

**Fecha:** a pendência **C1** do projeto (ação circular sob gradiente gaussiano) e
o critério de saída 2. A `NormalProjetada` entra na SPEC-06 como decisão medida, e
não como preferência de projeto.

**Não fecha:** (C). O crédito temporal no brinquedo sequencial permanece sem
diagnóstico, e vale registrar que a SPEC-06 roda num ambiente com recompensa mais
densa que o brinquedo (`R_t = (Σ visited − Ns)/Ns` muda a cada coleta), então o
modo de falha de (C) pode não transferir — mas isso é conjectura, não medição.

## Reprodução

```bash
uv run python scripts/spec05_circular.py --sementes 3 --tentativas 20000 --episodios 600
```

Artefato: `reports/spec05-circular.json`. Semente do bootstrap fixa em `20260809`
via `ArvoreSementes`; os 400 alvos de avaliação vêm de semente fixa `999`, e é isso
que autoriza o pareamento.
