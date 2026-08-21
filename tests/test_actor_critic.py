"""Política gaussiana (§4.3.3) e Algoritmo 4.4.

Diferente das redes, que têm oráculo externo, o laço do Algoritmo 4.4 é
transcrição do pseudocódigo da página 48. Validá-lo diferencialmente exigiria
rodar o agente Julia inteiro, que está amarrado aos tipos de ambiente e buffer
do `RLWorktable.jl`. O que se faz aqui é diferente e mais fraco, e vale dizer:
os invariantes estruturais do pseudocódigo são verificados um a um, e um teste
de aprendizado fecha a ponta — se o sinal de vantagem estivesse trocado ou o
retorno montado ao contrário, a política não convergiria.
"""

from __future__ import annotations

import numpy as np
import pytest

from uavwnn.rl import AtorCriticoNTupla, PoliticaGaussiana
from uavwnn.wnn import RedeGradienteFuncional, RedeRegressaoEsquecimento

B, TUPLA = 64, 8


def _agente(sigma=0.25, alpha=0.02, m=1, gamma=1.0, phi=1.0, saidas=1, seed=0):
    ator = RedeGradienteFuncional(B, TUPLA, n_saidas=saidas, seed=seed)
    critico = RedeRegressaoEsquecimento(B, TUPLA, n_saidas=1, phi=phi, seed=seed)
    return AtorCriticoNTupla(
        politica=PoliticaGaussiana(ator, escala_covariancia=sigma),
        critico=critico, passos=m, desconto=gamma, taxa=alpha,
        rng=np.random.default_rng(seed),
    )


# --- política: eqs. (4.34) e (4.35) ---------------------------------------


def test_sigma_e_covariancia_nao_desvio():
    """A eq. (4.34) diz "predefined **covariance** matrix", então o desvio da
    amostragem é `√Σ`. O `ContinousActionPolicy.jl` amostra com Σ direto —
    divergência registrada como P13, disponível via `escala_como_desvio`."""
    rede = RedeGradienteFuncional(B, TUPLA, n_saidas=1, seed=0)
    artigo = PoliticaGaussiana(rede, escala_covariancia=0.25)
    julia = PoliticaGaussiana(rede, escala_covariancia=0.25, escala_como_desvio=True)
    assert artigo.desvio == pytest.approx(0.5)
    assert julia.desvio == pytest.approx(0.25)
    assert artigo.inv_covariancia == julia.inv_covariancia == pytest.approx(4.0)


def test_amostragem_tem_a_dispersao_pedida():
    rede = RedeGradienteFuncional(B, TUPLA, n_saidas=1, seed=0)
    pol = PoliticaGaussiana(rede, escala_covariancia=0.25)
    bits = np.random.default_rng(1).random((1, B)) > 0.5
    rng = np.random.default_rng(0)
    amostras = np.array([pol.amostrar(bits, rng)[0] for _ in range(4000)])
    assert amostras.mean() == pytest.approx(0.0, abs=0.05)
    assert amostras.std() == pytest.approx(0.5, rel=0.05)


def test_grad_log_e_sigma_inversa_vezes_o_residuo():
    """Eq. (4.35): `Σ⁻¹ (a − f(s))`."""
    rede = RedeGradienteFuncional(B, TUPLA, n_saidas=2, seed=0)
    pol = PoliticaGaussiana(rede, escala_covariancia=0.5)
    bits = np.random.default_rng(1).random((1, B)) > 0.5
    rede.atualizar(bits, np.array([0.3, -0.2]))
    acao = np.array([1.0, 1.0])
    esperado = 2.0 * (acao - rede.prever(bits)[0])
    assert np.allclose(pol.grad_log(bits, acao), esperado, atol=1e-12)


def test_covariancia_nao_positiva_falha_cedo():
    rede = RedeGradienteFuncional(B, TUPLA, seed=0)
    with pytest.raises(ValueError, match="deve ser > 0"):
        PoliticaGaussiana(rede, escala_covariancia=0.0)


# --- Algoritmo 4.4: invariantes estruturais -------------------------------


def test_retorno_de_m_passos_com_bootstrap():
    """`G = R₁ + γR₂ + … + γ^{m−1}R_m + γ^m V(S_{τ+m})`."""
    ag = _agente(m=3, gamma=0.9)
    bits = np.random.default_rng(0).random((4, B)) > 0.5
    for i, r in enumerate((1.0, 2.0, 3.0)):
        ag.registrar(bits[i:i + 1], np.array([0.0]), r)
    g = ag._retorno(bootstrap=10.0)
    assert g == pytest.approx(1.0 + 0.9 * 2.0 + 0.81 * 3.0 + 0.729 * 10.0)


def test_vantagem_usa_o_valor_antes_do_treino_do_critico():
    """O pseudocódigo calcula δ com a rede no estado corrente e só depois
    aplica a eq. (4.8). Se a ordem se invertesse, ator e crítico passariam a
    enxergar estimativas diferentes do mesmo estado."""
    ag = _agente(m=1, gamma=1.0, phi=1.0)
    bits = (np.random.default_rng(0).random((1, B)) > 0.5)
    ag.critico.treinar(bits, np.array([2.0]))
    valor_antes = float(ag.critico.prever(bits)[0, 0])

    vistos = []
    original = ag.politica.rede.atualizar

    def espiao(b, passo):
        vistos.append(passo.copy())
        original(b, passo)

    ag.politica.rede.atualizar = espiao
    ag.registrar(bits, np.array([1.0]), 6.0)
    ag.fim_de_episodio()

    delta_esperado = 6.0 - valor_antes
    grad = ag.politica.inv_covariancia * (1.0 - 0.0)   # f(s) ainda era zero
    assert vistos[0][0] == pytest.approx(ag.taxa * delta_esperado * grad, rel=1e-9)
    assert float(ag.critico.prever(bits)[0, 0]) == pytest.approx(4.0)


def test_buffer_so_consome_quando_enche():
    ag = _agente(m=3)
    bits = np.random.default_rng(0).random((5, B)) > 0.5
    for i in range(2):
        ag.registrar(bits[i:i + 1], np.array([0.0]), 1.0)
        ag.passo_intermediario(bits[i + 1:i + 2])
    assert len(ag._buffer) == 2, "não deveria consumir com o buffer curto"
    ag.registrar(bits[2:3], np.array([0.0]), 1.0)
    ag.passo_intermediario(bits[3:4])
    assert len(ag._buffer) == 2, "deveria ter consumido exatamente uma"


def test_fim_de_episodio_esvazia_sem_bootstrap():
    ag = _agente(m=5, gamma=1.0)
    bits = np.random.default_rng(0).random((3, B)) > 0.5
    for i in range(3):
        ag.registrar(bits[i:i + 1], np.array([0.0]), 1.0)
    ag.fim_de_episodio()
    assert len(ag._buffer) == 0
    # A última transição tem G = 1 (nada depois dela); a primeira, G = 3.
    assert float(ag.critico.prever(bits[0:1])[0, 0]) == pytest.approx(3.0)
    assert float(ag.critico.prever(bits[2:3])[0, 0]) == pytest.approx(1.0)


def test_recozimento_da_taxa_e_opcional_e_nao_e_do_artigo():
    """O apêndice A lista α constante. O recozimento vem do código Julia."""
    assert _agente().taxa_corrente == pytest.approx(0.02)
    ag = _agente()
    ag.taxa_final, ag.decaimento_taxa = 0.0, 10
    ag._n_updates = 5
    assert ag.taxa_corrente == pytest.approx(0.01)
    ag._n_updates = 999
    assert ag.taxa_corrente == pytest.approx(0.0)


def test_parametros_invalidos_falham_cedo():
    with pytest.raises(ValueError, match="passos"):
        _agente(m=0)
    with pytest.raises(ValueError, match="desconto"):
        _agente(gamma=1.5)


# --- o laço aprende --------------------------------------------------------


def test_aprende_bandido_contextual_continuo():
    """Quatro contextos, ação escalar, recompensa `−(a − alvo(s))²`.

    A política ótima emite `alvo(s)`. É o teste que fecha a ponta: com o sinal
    da vantagem trocado, ou o retorno montado na ordem errada, a média da
    política se afastaria dos alvos em vez de convergir.

    `α = 0,002` não é arbitrário. Varrendo α em 5 sementes, o erro cresce de
    forma **monótona** com ele — 0,044 em 0,002 contra 0,238 em 0,05 —, e em
    α = 0,05 uma das sementes chega a divergir. É a instabilidade de passo fixo
    que a própria dissertação aponta como limitação (cap. 6) ao sugerir
    TRPO/PPO. Registrado no relatório da SPEC-05.
    """
    rng = np.random.default_rng(7)
    contextos = rng.random((4, B)) > 0.5
    alvos = np.array([-1.0, -0.3, 0.3, 1.0])

    ag = _agente(sigma=0.25, alpha=0.002, m=1, gamma=1.0, phi=0.99, seed=3)
    erro_inicial = np.abs(
        np.array([ag.politica.media(contextos[i:i + 1])[0] for i in range(4)]) - alvos
    ).mean()

    for passo in range(20_000):
        i = passo % 4
        s = contextos[i:i + 1]
        a = ag.agir(s)
        ag.registrar(s, a, -float((a[0] - alvos[i]) ** 2))
        ag.fim_de_episodio()

    final = np.array([ag.politica.media(contextos[i:i + 1])[0] for i in range(4)])
    erro_final = np.abs(final - alvos).mean()

    assert erro_final < 0.08, f"não convergiu: {final} contra {alvos}"
    assert erro_final < erro_inicial / 5
    # Acertar a média não basta: a ordem dos contextos tem de sobreviver.
    assert np.all(np.diff(final) > 0), f"perdeu a ordenação: {final}"


def test_o_critico_estima_o_valor_do_estado():
    """No bandido de um passo, `V(s) = E[−(a − alvo)²] = −Σ` na política ótima.

    Verifica o outro lado do laço: se o crítico estivesse só acumulando sem
    normalizar pelos pesos da eq. (4.10), a escala sairia errada.
    """
    rng = np.random.default_rng(11)
    contextos = rng.random((3, B)) > 0.5
    alvos = np.array([-0.5, 0.0, 0.5])
    sigma = 0.25

    ag = _agente(sigma=sigma, alpha=0.002, m=1, gamma=1.0, phi=0.99, seed=5)
    for passo in range(20_000):
        i = passo % 3
        s = contextos[i:i + 1]
        a = ag.agir(s)
        ag.registrar(s, a, -float((a[0] - alvos[i]) ** 2))
        ag.fim_de_episodio()

    valores = np.array([float(ag.critico.prever(contextos[i:i + 1])[0, 0])
                        for i in range(3)])
    assert np.all(valores < 0), f"valor deveria ser negativo: {valores}"
    assert np.abs(valores - (-sigma)).max() < 0.15, f"escala errada: {valores}"


# ---------------------------------------------------------------------------
# regressão: o aprendizado tem de ATRAVESSAR episódios
# ---------------------------------------------------------------------------


def test_aprendizado_sobrevive_a_troca_de_episodio():
    """A ocupação de memória só pode **crescer** ao longo do treino.

    ESTE TESTE EXISTE PORQUE A AUSÊNCIA DELE CUSTOU UMA REPRODUÇÃO INTEIRA.

    O `scripts/spec05_mountaincar.py` chamava `reiniciar()` no começo de cada
    episódio, acreditando que ela só esvaziava a janela de `m` passos. Ela apaga as
    duas redes. O efeito foi treinar 2000 episódios **independentes**, cada um
    partindo do zero — e o resultado (retorno ≈ 0, o ótimo local de "não mover o
    carro") era indistinguível de "o método não funciona". Passei um bom tempo
    diagnosticando o método antes de olhar o meu próprio laço.

    O sintoma que denunciou é o que este teste verifica: a ocupação **oscilava**
    nas centenas (188 → 722 → 283 → …) quando um dicionário que só recebe chaves
    não pode encolher.
    """
    import numpy as np

    from uavwnn.rl.actor_critic import AtorCriticoNTupla
    from uavwnn.rl.politica import PoliticaGaussiana
    from uavwnn.wnn.redes import RedeGradienteFuncional, RedeRegressaoEsquecimento

    n_bits, tupla = 64, 8
    ator = RedeGradienteFuncional(n_bits, tupla, n_saidas=1, seed=0, c_max=None)
    critico = RedeRegressaoEsquecimento(n_bits, tupla, n_saidas=1, phi=0.85, seed=0)
    ag = AtorCriticoNTupla(politica=PoliticaGaussiana(ator, 0.5), critico=critico,
                           passos=4, desconto=1.0, taxa=1e-3,
                           rng=np.random.default_rng(0))

    g = np.random.default_rng(1)
    ocupacoes = []
    for _ in range(6):
        ag.novo_episodio()                      # a chamada CERTA de início
        for _ in range(10):
            bits = g.random(n_bits) > 0.5
            acao = ag.agir(bits)
            ag.registrar(bits, acao, float(g.normal()))
            ag.passo_intermediario(g.random(n_bits) > 0.5)
        ag.fim_de_episodio()
        ocupacoes.append(ator.entradas_ocupadas())

    assert ocupacoes == sorted(ocupacoes), (
        f"a ocupação encolheu entre episódios: {ocupacoes}. Alguma coisa está "
        f"apagando a rede no meio do treino")
    assert ocupacoes[-1] > ocupacoes[0], (
        f"a ocupação não cresceu em 6 episódios: {ocupacoes}")


def test_reiniciar_apaga_mesmo_e_isso_e_intencional():
    """A contraparte: `reiniciar` **deve** zerar, e o contrato fica fixado.

    Sem este teste, alguém poderia "consertar" o `reiniciar` para não apagar — e
    aí a operação de recomeçar uma execução deixaria de existir.
    """
    import numpy as np

    from uavwnn.rl.actor_critic import AtorCriticoNTupla
    from uavwnn.rl.politica import PoliticaGaussiana
    from uavwnn.wnn.redes import RedeGradienteFuncional, RedeRegressaoEsquecimento

    ator = RedeGradienteFuncional(64, 8, n_saidas=1, seed=0, c_max=None)
    critico = RedeRegressaoEsquecimento(64, 8, n_saidas=1, phi=0.85, seed=0)
    ag = AtorCriticoNTupla(politica=PoliticaGaussiana(ator, 0.5), critico=critico,
                           passos=4, desconto=1.0, taxa=1e-3,
                           rng=np.random.default_rng(0))
    g = np.random.default_rng(2)
    for _ in range(10):
        bits = g.random(64) > 0.5
        ag.registrar(bits, ag.agir(bits), float(g.normal()))
    ag.fim_de_episodio()
    assert ator.entradas_ocupadas() > 0

    ag.reiniciar()
    assert ator.entradas_ocupadas() == 0
    assert critico.entradas_ocupadas() == 0
