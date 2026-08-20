"""As identidades matemáticas da dissertação, verificadas no nosso código.

Os testes de `test_redes.py` provam que reproduzimos o `Ramnet.jl` bit a bit.
Isso é forte, mas prova **equivalência com uma implementação**, não com a
matemática — se o Katopodis e nós tivéssemos o mesmo mal-entendido, os dois
concordariam. Este arquivo fecha esse flanco: cada teste verifica uma equação
numerada do texto.

E não é preciosismo teórico. O PRD justifica a rede n-tuple como *"tile coding
com ladrilhamento aleatório, e é por isso que herda as garantias de convergência
de TD com aproximação linear"*. Essa herança **depende** de a rede ser de fato
uma máquina de núcleo, que é o que as eqs. (3.22) e (4.16) provam. Nunca havia
sido testado.

Cobertura por equação:

| eq. | o que afirma |
|---|---|
| (3.17) | distância de tupla `δ(x,z) = Σ_i ⟦τ_i(x) ≠ τ_i(z)⟧` |
| (3.19) | `V[i](x) = Σ_t y_t M[i]_t(x)` |
| (3.20) | `C[i](x) = Σ_t M[i]_t(x)` |
| (3.21) | `Σ_i M[i]_t(x) = N(1 − δ/N)` |
| (3.22) | a rede de regressão **é** um estimador de Nadaraya-Watson |
| (3.23) | termômetro |
| (3.24) | circular |
| (4.9)/(4.10) | limite `φ → 1` recupera a rede original |
| (4.13) | `v[i]_t(x) = Σ_j α_j M[i]_j(x)` |
| (4.16) | a rede de gradiente funcional **é** uma máquina de núcleo |
"""

from __future__ import annotations

import numpy as np
import pytest

from uavwnn.encoding.artigo import CircularArtigo, TermometroArtigo
from uavwnn.wnn import RedeGradienteFuncional, RedeRegressaoEsquecimento

B, TUPLA = 48, 6


# --- utilitários que traduzem as definições do texto ----------------------


def tuplas(rede, x):
    """`τ[i](x)` — o endereço que `x` indexa em cada neurônio."""
    return rede.enderecos(np.atleast_2d(x))[0]


def distancia_de_tupla(rede, x, z) -> int:
    """Eq. (3.17), literal."""
    return int((tuplas(rede, x) != tuplas(rede, z)).sum())


def nucleo(rede, x, z) -> float:
    """`k(x,z) = 1 − δ(x,z)/N`, definido em (3.21)/(3.22)."""
    return 1.0 - distancia_de_tupla(rede, x, z) / rede.n_ram


def indicadores(rede, x, z):
    """`M[i](x,z)` da eq. (3.18), como vetor sobre os neurônios."""
    return (tuplas(rede, x) == tuplas(rede, z)).astype(np.int64)


# --- (3.17), (3.18), (3.21): distância de tupla e o núcleo implícito ------


def test_distancia_de_tupla_e_zero_sse_padroes_identicos():
    """"If the two patterns are identical, all their tuples are the same, and
    their tuple distance is zero." — §3.2.2"""
    rede = RedeGradienteFuncional(B, TUPLA, seed=0)
    rng = np.random.default_rng(0)
    x = rng.random(B) > 0.5
    assert distancia_de_tupla(rede, x, x) == 0
    assert nucleo(rede, x, x) == 1.0

    # E é estritamente positiva para padrões que diferem em algum bit lido.
    z = x.copy()
    z[rede.mapeamento[0][0]] = ~z[rede.mapeamento[0][0]]
    assert distancia_de_tupla(rede, x, z) > 0


def test_soma_dos_indicadores_e_N_vezes_o_nucleo():
    """Eq. (3.21): `Σ_i M[i]_t(x) = N(1 − δ(x,x_t)/N)`."""
    rede = RedeGradienteFuncional(B, TUPLA, seed=1)
    rng = np.random.default_rng(2)
    for _ in range(40):
        x, z = rng.random(B) > 0.5, rng.random(B) > 0.5
        soma = int(indicadores(rede, x, z).sum())
        assert soma == rede.n_ram - distancia_de_tupla(rede, x, z)
        assert soma == pytest.approx(rede.n_ram * nucleo(rede, x, z))


def test_nucleo_vive_em_zero_um_e_decai_com_a_dissimilaridade():
    """`k` é similaridade: 1 no próprio ponto, e não cresce quando se afasta.

    O passeio inverte um bit lido por vez, então a distância de tupla não pode
    diminuir — é o que faz de `k` uma função de similaridade utilizável.
    """
    rede = RedeGradienteFuncional(B, TUPLA, seed=3)
    x = np.zeros(B, dtype=bool)
    anterior, vistos = 1.0, []
    for tupla in rede.mapeamento:                  # inverte um bit de cada RAM
        x = x.copy()
        x[tupla[0]] = True
        k = nucleo(rede, np.zeros(B, dtype=bool), x)
        assert 0.0 <= k <= 1.0
        assert k <= anterior
        anterior = k
        vistos.append(k)
    assert vistos[-1] < 1.0, "o passeio deveria ter reduzido a similaridade"


# --- (3.19), (3.20), (3.22): a rede de regressão é Nadaraya-Watson --------


def test_valor_e_contador_sao_somas_sobre_o_conjunto():
    """Eqs. (3.19) e (3.20): `V[i](x) = Σ_t y_t M[i]_t(x)`, `C[i](x) = Σ_t M[i]_t(x)`.

    Com `φ = 1` a eq. (4.8) degenera exatamente na regra da (3.15), então a
    `RedeRegressaoEsquecimento` é a rede original e estas somas devem fechar.
    """
    rede = RedeRegressaoEsquecimento(B, TUPLA, phi=1.0, seed=4)
    rng = np.random.default_rng(5)
    X = rng.random((60, B)) > 0.5
    Y = rng.normal(size=60)
    for t in range(len(X)):
        rede.treinar(X[t:t + 1], np.array([Y[t]]))

    consulta = rng.random(B) > 0.5
    end = tuplas(rede, consulta)
    for i in range(rede.n_ram):
        M = np.array([1 if tuplas(rede, X[t])[i] == end[i] else 0
                      for t in range(len(X))])
        item = rede.memoria[i].get(int(end[i]))
        v = 0.0 if item is None else float(item[0][0])
        c = 0.0 if item is None else float(item[1][0])
        assert v == pytest.approx(float((Y * M).sum()), abs=1e-9)   # (3.19)
        assert c == pytest.approx(float(M.sum()), abs=1e-9)         # (3.20)


def test_rede_de_regressao_e_estimador_de_nadaraya_watson():
    """Eq. (3.22) — a identidade central do capítulo 3:

        f(x) = Σ_t y_t k(x, x_t) / Σ_t k(x, x_t),   k = 1 − δ/N

    É o que autoriza tratar a rede como aproximação linear com base de núcleo,
    e portanto o que sustenta a alegação de convergência do PRD.
    """
    rede = RedeRegressaoEsquecimento(B, TUPLA, phi=1.0, seed=6)
    rng = np.random.default_rng(7)
    X = rng.random((80, B)) > 0.5
    Y = rng.normal(size=80)
    for t in range(len(X)):
        rede.treinar(X[t:t + 1], np.array([Y[t]]))

    for consulta in rng.random((15, B)) > 0.5:
        ks = np.array([nucleo(rede, consulta, X[t]) for t in range(len(X))])
        obtido = float(rede.prever(consulta[None, :])[0, 0])
        if ks.sum() == 0:
            assert obtido == 0.0, "eq. (3.16): denominador nulo -> 0"
        else:
            esperado = float((Y * ks).sum() / ks.sum())
            assert obtido == pytest.approx(esperado, abs=1e-9)


# --- (4.13), (4.16): a rede de gradiente funcional é máquina de núcleo ----


def test_memoria_do_ator_e_soma_das_atualizacoes():
    """Eq. (4.13): `v[i]_t(x) = Σ_j α_j M[i]_j(x)`."""
    rede = RedeGradienteFuncional(B, TUPLA, n_saidas=1, seed=8)
    rng = np.random.default_rng(9)
    X = rng.random((50, B)) > 0.5
    alfas = rng.normal(size=50)
    for j in range(len(X)):
        rede.atualizar(X[j:j + 1], np.array([alfas[j]]))

    consulta = rng.random(B) > 0.5
    end = tuplas(rede, consulta)
    for i in range(rede.n_ram):
        M = np.array([1 if tuplas(rede, X[j])[i] == end[i] else 0
                      for j in range(len(X))])
        item = rede.memoria[i].get(int(end[i]))
        v = 0.0 if item is None else float(item[0])
        assert v == pytest.approx(float((alfas * M).sum()), abs=1e-9)


def test_rede_de_gradiente_funcional_e_maquina_de_nucleo():
    """Eq. (4.16) — a identidade central do §4.3.1:

        f_t(x) = Σ_j α_j k(x, x_j)

    Note a diferença crucial em relação à (3.22): aqui **não há normalização**.
    A rede de regressão é uma média ponderada de núcleo; a de gradiente
    funcional é uma soma de funções de base. É o que permite a ela representar
    funções que a outra não representa, e é a razão de ser o modelo do ator.
    """
    rede = RedeGradienteFuncional(B, TUPLA, n_saidas=2, seed=10)
    rng = np.random.default_rng(11)
    X = rng.random((60, B)) > 0.5
    alfas = rng.normal(size=(60, 2))
    for j in range(len(X)):
        rede.atualizar(X[j:j + 1], alfas[j])

    for consulta in rng.random((15, B)) > 0.5:
        ks = np.array([nucleo(rede, consulta, X[j]) for j in range(len(X))])
        esperado = (alfas * ks[:, None]).sum(axis=0)
        obtido = rede.prever(consulta[None, :])[0]
        assert np.allclose(obtido, esperado, atol=1e-9)


def test_as_duas_redes_diferem_exatamente_pela_normalizacao():
    """Contraste explícito entre (3.22) e (4.16), sobre os mesmos dados.

    Se alguém "simplificasse" as duas para a mesma fórmula, este teste cai.
    """
    rng = np.random.default_rng(12)
    X = rng.random((40, B)) > 0.5
    Y = rng.normal(size=40)

    ator = RedeGradienteFuncional(B, TUPLA, n_saidas=1, seed=13)
    critico = RedeRegressaoEsquecimento(B, TUPLA, phi=1.0, seed=13)
    for t in range(len(X)):
        ator.atualizar(X[t:t + 1], np.array([Y[t]]))
        critico.treinar(X[t:t + 1], np.array([Y[t]]))

    consulta = X[0][None, :]
    ks = np.array([nucleo(ator, X[0], X[t]) for t in range(len(X))])
    assert float(ator.prever(consulta)[0, 0]) == pytest.approx(
        float((Y * ks).sum()), abs=1e-9)
    assert float(critico.prever(consulta)[0, 0]) == pytest.approx(
        float((Y * ks).sum() / ks.sum()), abs=1e-9)


# --- (4.9)/(4.10): o limite φ → 1 -----------------------------------------


def test_esquecimento_tende_a_rede_original_quando_phi_vai_a_um():
    """A §4.2.1 apresenta a rede com esquecimento como generalização da (3.16).

    Verificamos a continuidade: com `φ = 1 − ε` a saída converge para a média
    simples, e o desvio encolhe com ε.
    """
    rng = np.random.default_rng(14)
    X = rng.random((30, B)) > 0.5
    Y = rng.normal(size=30)

    def treina(phi):
        r = RedeRegressaoEsquecimento(B, TUPLA, phi=phi, seed=15)
        for t in range(len(X)):
            r.treinar(X[t:t + 1], np.array([Y[t]]))
        return r.prever(X[:10])

    referencia = treina(1.0)
    desvios = [np.abs(treina(1 - eps) - referencia).max()
               for eps in (1e-2, 1e-4, 1e-6)]
    assert desvios[0] > desvios[1] > desvios[2]
    assert desvios[-1] < 1e-4


# --- (3.23), (3.24): os codificadores do texto ---------------------------


def test_termometro_do_artigo_segue_a_eq_3_23():
    """`S_i = 1` se `x > l + (i−1)(u−l)/R`, com `>` estrito."""
    enc = TermometroArtigo(0.0, 1.0, resolucao=5)
    assert np.allclose(enc.limiares, [0.0, 0.2, 0.4, 0.6, 0.8])

    # x = 0.0 está SOBRE o primeiro limiar; com `>` estrito, nada acende.
    assert enc.codificar(np.array([[0.0]])).sum() == 0
    assert list(enc.codificar(np.array([[0.5]]))[0]) == [1, 1, 1, 0, 0]
    assert enc.codificar(np.array([[1.0]])).sum() == 5

    # monotonicidade: mais valor, nunca menos bits
    vs = np.linspace(0, 1, 41)[:, None]
    contagens = enc.codificar(vs).sum(axis=1)
    assert np.all(np.diff(contagens) >= 0)


def test_circular_do_artigo_segue_a_eq_3_24():
    """`n_shift = ⌊(x−l)/(u−l)·R⌋`, padrão base com `⌊R/2⌋` bits."""
    enc = CircularArtigo(0.0, 1.0, resolucao=8)
    assert enc.bits_ativos == 4
    assert list(enc.base.astype(int)) == [1, 1, 1, 1, 0, 0, 0, 0]
    assert enc.deslocamento(np.array([0.0, 0.25, 0.5, 0.99])).tolist() == [0, 2, 4, 7]

    # número de bits ativos é CONSTANTE — a diferença estrutural em relação ao
    # termômetro, e o que quebra a nossa convenção de padding todo-zero.
    vs = np.linspace(0, 0.999, 60)[:, None]
    assert set(enc.codificar(vs).sum(axis=1).tolist()) == {4}


def test_circular_faz_wrap_e_o_termometro_nao():
    """"the circular behavior favors a radial similarity, as extreme values are
    mapped to similar patterns" — §3.2.3.

    É a propriedade que o termômetro não tem, e a razão de o apêndice A usar
    circular em CartPole e Hopper.
    """
    R = 16
    circ, term = CircularArtigo(0.0, 1.0, R), TermometroArtigo(0.0, 1.0, R)
    baixo, alto = np.array([[0.0]]), np.array([[0.999]])

    def hamming(enc):
        return int((enc.codificar(baixo) != enc.codificar(alto)).sum())

    assert hamming(circ) == 2, "extremos deveriam ser vizinhos no círculo"
    assert hamming(term) == R, "no termômetro, extremos são opostos"

    # E o meio do intervalo é o ponto mais distante de 0 no circular.
    meio = np.array([[0.5]])
    assert int((circ.codificar(baixo) != circ.codificar(meio)).sum()) == R


# --- §3.2.4: o trade-off complexidade / generalização --------------------


def _fracao_sem_resposta(tupla, n_bits, n_treino=120, seed=0):
    """Fração de consultas novas que endereçam SÓ posições nunca escritas.

    É o "unable to generalize" de §3.2.4: a rede devolve a resposta padrão.
    """
    rng = np.random.default_rng(seed)
    rede = RedeGradienteFuncional(n_bits, tupla, n_saidas=1, seed=seed)
    X = rng.random((n_treino, n_bits)) > 0.5
    for t in range(n_treino):
        rede.atualizar(X[t:t + 1], np.array([1.0]))
    novas = rng.random((200, n_bits)) > 0.5
    return float((rede.prever(novas)[:, 0] == 0.0).mean())


def test_tupla_maior_generaliza_pior():
    """"Larger tuples ... it's not unlikely that a novel input will address
    positions in all neurons that weren't accessed during training. In a
    situation such as this, the model is unable to generalize." — §3.2.4

    Este teste é o que dá conteúdo empírico à P11: a dissertação usa `n` de 16
    a 64, mas com `B` muito maior (CartPole: 4 obs × 128 = 512 bits). O que
    governa a generalização é a razão `n/B`, não `n` sozinho — e é por isso que
    copiar o `n` dela sem olhar o nosso `B` seria erro.
    """
    n_bits = 96
    fracoes = {t: _fracao_sem_resposta(t, n_bits) for t in (4, 8, 16, 32)}
    valores = [fracoes[t] for t in (4, 8, 16, 32)]
    assert valores == sorted(valores), f"deveria crescer com a tupla: {fracoes}"
    assert fracoes[4] < 0.1, f"tupla pequena deveria generalizar: {fracoes[4]}"
    assert fracoes[32] > 0.5, f"tupla grande deveria falhar: {fracoes[32]}"


def test_ram_unica_memoriza_e_nao_generaliza():
    """"The degenerate case ... a network made up of a single RAM neuron. Such a
    model is able to memorize all training samples, but can only produce a
    default response to previously unseen data." — §3.2.4"""
    n_bits = 20
    rede = RedeGradienteFuncional(n_bits, n_bits, n_saidas=1, seed=0)   # 1 RAM
    assert rede.n_ram == 1

    rng = np.random.default_rng(1)
    X = rng.random((25, n_bits)) > 0.5
    for t in range(len(X)):
        rede.atualizar(X[t:t + 1], np.array([float(t + 1)]))

    # memoriza: cada treino recupera exatamente o seu valor
    assert np.allclose(rede.prever(X)[:, 0], np.arange(1, 26))
    # e só devolve o padrão para o que não viu
    novas = rng.random((60, n_bits)) > 0.5
    assert (rede.prever(novas)[:, 0] == 0.0).mean() > 0.95


def test_resolucao_menor_alia_mais():
    """"making use of a smaller encoding resolution allows the model to
    generalize ... as doing so entails a greater amount of aliasing" — §3.2.4

    Aliasing medido como o próprio texto define: entradas distintas mapeadas
    para padrões binários idênticos.
    """
    vs = np.linspace(0.0, 0.999, 300)[:, None]
    distintos = {R: len(np.unique(TermometroArtigo(0.0, 1.0, R).codificar(vs),
                                  axis=0))
                 for R in (4, 8, 16, 64)}
    valores = [distintos[R] for R in (4, 8, 16, 64)]
    assert valores == sorted(valores), f"deveria crescer com R: {distintos}"
    # Um termômetro de resolução R tem exatamente R+1 palavras-código
    # (0000, 1000, 1100, 1110, 1111 para R = 4), então esse é o teto do número
    # de padrões distintos e a medida exata do aliasing que R impõe.
    for R, n in distintos.items():
        assert n <= R + 1, f"R={R} produziu {n} padrões, teto é {R + 1}"
    assert distintos[4] == 5, "com 300 entradas, R=4 deve saturar as 5 palavras"
    assert distintos[64] > 50


def test_phi_zero_guarda_so_a_observacao_mais_recente():
    """"[φ] ranges from 0, where only the most recent observation to address a
    position influence it, to 1, where all samples ... influence its content
    equally and the system behaves exactly as an n-tuple regression network."
    — §5.2.3

    Os dois extremos do intervalo declarado. `φ = 1` já está coberto pelas
    identidades de Nadaraya-Watson; aqui fecha-se o outro lado.

    Com `φ = 0` a eq. (4.8) vira `v ← y`, e a (4.10) dá `s = (1−0^c)/(1−0) = 1`
    para `c ≥ 1`. Logo `f = Σ_i y_último[i] / N`.
    """
    rede = RedeRegressaoEsquecimento(B, TUPLA, phi=0.0, seed=16)
    bits = np.zeros((1, B), dtype=bool)
    bits[0, ::2] = True
    for y in (100.0, -7.0, 3.5):
        rede.treinar(bits, np.array([y]))
    # todas as RAMs foram endereçadas pelo mesmo padrão, então f = 3,5
    assert float(rede.prever(bits)[0, 0]) == pytest.approx(3.5, abs=1e-12)

    # e o histórico realmente não sobrevive: o valor guardado é só o último
    item = next(iter(rede.memoria[0].values()))
    assert float(item[0][0]) == pytest.approx(3.5, abs=1e-12)
    assert float(item[1][0]) == 3.0, "o contador ainda conta todas as visitas"
