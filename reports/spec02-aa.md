# Teste A/A — o bootstrap pareado está calibrado

SPEC-02 T-02.9 §2. Reprodução:

```bash
uv run python scripts/spec02_aa.py --execucoes 200
```

---

## 1. Por que este era o último bloqueio

Todo intervalo de confiança já reportado neste repositório passa por
`bootstrap_pareado` ou pelo mesmo procedimento escrito à mão:

| onde | o que dependia |
|---|---|
| SPEC-00 | validação 9/9 do guloso contra a fig. 7.3; `greedy` vs `smart` |
| SPEC-03 | comparação de custo pareada WNN vs MLP |
| T-B2 | replicação de §5.2.2 (empate num regime, vitória no outro) |
| SPEC-02 | validação 3/3 do MADDPG contra os valores publicados |

O critério de saída da spec é explícito: *"teste A/A com cobertura ≈ 95 % — **sem
isso, nenhum IC deste projeto é confiável**"*. Se a cobertura estivesse errada,
todos os quatro precisariam de revisão.

## 2. Resultado: calibrado

200 execuções × 500 episódios × `B = 1000`.

| fonte | reamostragem | cobertura | IC da cobertura | largura média do IC |
|---|---|---|---|---|
| sintética | **pareada** | **95,5 %** | [91,7 %; 97,6 %] | 0,983 |
| sintética | independente | 100,0 % | [98,1 %; 100,0 %] | **5,137** |
| campanha MADDPG | **pareada** | **93,5 %** | [89,2 %; 96,2 %] | 0,179 |
| campanha MADDPG | independente | 99,0 % | [96,4 %; 99,7 %] | 0,248 |

**O IC da cobertura contém 95 % nas duas fontes pareadas.** Com 200 execuções o
erro padrão da própria cobertura é ~1,5 pp, então "93,5 %" e "95,5 %" não são
distinguíveis de 95 % — e é por isso que a tolerância é o IC de Wilson e não uma
faixa escolhida a olho.

**Os ICs pareados já reportados no projeto se sustentam.**

## 3. O contra-exemplo falha, e é isso que dá valor ao teste

A linha `independente` reproduz o bug que o A/A existe para pegar: reamostrar os
dois métodos com **índices diferentes** em vez de aplicar o mesmo conjunto aos
dois.

O efeito é assimétrico e vale entender. A reamostragem independente **reintroduz a
variância entre cenários**, que o pareamento cancelaria. O IC fica largo demais —
**5,2× mais largo** no caso sintético — e a cobertura sobe para 100 %.

Ou seja: o bug não produz falsos positivos, produz **falsos negativos**. Diferenças
reais desapareceriam, e o MDE reportado estaria otimista. É o tipo de erro que
nunca dispara alarme, porque o resultado parece só "inconclusivo".

**Se o contra-exemplo passasse, o teste não estaria discriminando nada** — o script
checa isso e avisa.

## 4. O erro que eu cometi na primeira versão, porque ele é instrutivo

A primeira implementação comparava o vetor **contra si mesmo**. Resultado:

```
   sintetica        pareada     100.0%   [98.1%, 100.0%]         0.000
```

`Δ_b = média_b(x) − média_b(x) = 0` exatamente, em toda réplica. Largura de IC
**zero**, cobertura 100 %, e nenhuma informação. O teste passava sem testar.

O A/A correto precisa de **duas realizações independentes da mesma
distribuição**, e a spec já dizia como: *"`random_uniforme` contra ela mesma com
**sementes de política diferentes**"*. A estrutura tem duas componentes, e as
duas são necessárias:

```
efeito_cenario   COMPARTILHADO entre A e B   ← é o que o pareamento remove
ruido_politica   INDEPENDENTE                ← é o que sobra na diferença
```

Sem o efeito compartilhado não há pareamento a verificar. Sem o ruído
independente não há diferença a medir.

## 5. As duas fontes

**Sintética** — log-normal assimétrica **censurada** em `max_episode_length`. A
censura não é enfeite: `t_conc_pen` tem um **átomo** no extremo direito por
construção (D-02.2), e o bootstrap da média é sensível a átomos. Testar com
gaussianas esconderia exatamente o caso que nos interessa.

**Campanha MADDPG** — construída dos 900 runs reais. O CSV guarda, por run, a
média dos 200 episódios e o desvio **entre** episódios, o que dá as duas
componentes de que o A/A precisa:

```
efeito_cenario  = a média da run       (compartilhado)
ruido_politica  = desvio/√200          (independente, o erro da média)
```

É modelo, e declarado como tal — mas as duas escalas são **medidas**, não
inventadas.

## 6. Uma ressalva que fica

O A/A verifica a cobertura do procedimento **sob a hipótese nula**. Ele não
verifica poder, e não verifica cobertura sob efeito grande. O que existe para
isso é o MDE, que é coluna obrigatória em toda comparação — e a
`test_pareamento_da_ic_muito_mais_estreito_que_independencia`, que constrói um
efeito real de 2 s e confirma que o pareado o detecta e o independente não.

Também vale registrar: **o IC mede a incerteza do cenário, não a do treino**. Um
IC bootstrap sobre 500 episódios de **um** modelo treinado não diz nada sobre a
variabilidade entre execuções de treino. A SPEC-06 roda ≥ 3 sementes de treino e
reporta a dispersão entre elas **separadamente** — é o erro mais comum na
literatura de RL e não vamos cometê-lo por acidente.

## 7. Na suíte

`tests/test_eval_stats.py` traz uma versão reduzida do A/A (60 execuções, `B =
400`, faixa larga de propósito) para que uma regressão no pareamento apareça na
suíte, e não só quando alguém lembrar de rodar o script. Mais 26 testes de
propriedade: monotonicidade de `t_conc_pen`, LB-1 como limite válido, disjunção
das sementes, Gini, Wilcoxon/McNemar degenerados, Holm monótono, e o caso
construído em que **ICs marginais se sobrepõem e a diferença pareada é
conclusiva**.
