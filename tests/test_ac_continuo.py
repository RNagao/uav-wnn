"""Actor-Critic contínuo pareado — SPEC-05 §4.3, §4.4, §4.8 e T-05.3.

Três coisas aqui podem quebrar em silêncio, e é por elas que os testes existem:

1. **Retropropagação da MLP escrita à mão.** O modo de falha é perverso: um
   baseline artificialmente fraco **infla H1**, empurrando o resultado na direção
   que nos favorece. Guarda: diferenças finitas em cada parâmetro.
2. **`amostrado ≠ executado`.** Usar a ação recortada no resíduo `(z − f)` viesa
   o gradiente exatamente onde a política pede velocidade máxima — a região onde
   ela passa a maior parte do tempo.
3. **Sinal do gradiente na MLP.** Adam desce; a ascensão precisa do negativo.
   Trocar faz a política fugir do que funciona.
"""

from __future__ import annotations

import numpy as np
import pytest

from uavwnn.agents import (
    D_ATOR,
    MLP,
    Acao,
    Adam,
    AtorCriticoContinuo,
    ContextoEpisodio,
    MlpAprox,
    NTuplaDissertacaoAprox,
    NormalProjetada,
    Observacao,
    Policy,
    V_MAX,
)
from uavwnn.encoding.artigo import TermometroArtigo


def _ctx():
    return ContextoEpisodio(num_drones=1, num_sensores=12, scenario_size=100.0,
                            communication_range=20.0, dt=0.5)


def _obs(vetor):
    return Observacao(vetor=np.asarray(vetor, float), p_a=np.zeros((0, 2)),
                      p_s=np.zeros((12, 2)), valido_s=np.zeros(12, bool),
                      valido_a=np.zeros(0, bool), k=0, t=0.0)


# --- §4.3: a normal projetada resolve o wrap-around ---------------------


def test_saida_do_ator_tem_tres_componentes():
    """`(u₁, u₂, μ_s)`. A tabela do PRD §5.5 diz `→2` e está errada — são +33
    parâmetros, desprezível no total, mas o número publicado tem de ser o certo."""
    assert D_ATOR == 3
    assert MLP(25, D_ATOR).dims[-1] == 3


@pytest.mark.parametrize("z,graus", [
    ((1.0, 0.0, 0.5), 0.0),
    ((0.0, 1.0, 0.5), 90.0),
    ((-1.0, 0.0, 0.5), 180.0),
    ((0.0, -1.0, 0.5), 270.0),
])
def test_atan2_cobre_o_circulo_inteiro(z, graus):
    a = NormalProjetada.para_acao(np.array(z))
    assert a.theta_norm * 360.0 == pytest.approx(graus, abs=1e-6)


def test_wrap_around_nao_tem_descontinuidade():
    """**É isto que resolve a C1.**

    Um `z₂` levemente negativo tem de virar rumo perto de 360°, não um valor
    negativo que estouraria a faixa `[0,1]` da ação. Com gaussiana direta sobre
    `θ` isso exigiria correção de fase; aqui sai do `atan2`.
    """
    a = NormalProjetada.para_acao(np.array([1.0, -1e-6, 0.5]))
    assert 0.999 < a.theta_norm <= 1.0
    b = NormalProjetada.para_acao(np.array([1.0, +1e-6, 0.5]))
    assert 0.0 <= b.theta_norm < 0.001
    # e os dois são vizinhos no círculo, apesar de opostos em `theta_norm`
    d = min(abs(a.theta_norm - b.theta_norm), 1 - abs(a.theta_norm - b.theta_norm))
    assert d < 1e-5


def test_amostrado_diferente_de_executado():
    """A regra que evita o bug mais provável da spec.

    `amostrar` devolve os dois: a ação **recortada** que vai ao ambiente e o `z`
    **bruto** que vai ao gradiente. Aqui `z₃ = 1,8` é recortado para `s = 1,0`,
    mas o resíduo tem de usar 1,8 — senão o sinal morre justamente onde a
    política pede velocidade máxima.
    """
    p = NormalProjetada(sigma=0.1)
    f = np.array([1.0, 0.0, 0.9])
    acao, z = p.amostrar(f, np.random.default_rng(0))

    # A ação é recortada em [0,1]; o z bruto pode passar de 1.
    assert 0.0 <= acao.s_norm <= 1.0
    z_alto = np.array([z[0], z[1], 1.8])
    assert NormalProjetada.para_acao(z_alto).s_norm == pytest.approx(1.0)

    # E é AQUI que os dois caminhos divergem: substituir o valor recortado no
    # resíduo mata o sinal justamente onde a política pede velocidade máxima.
    certo = p.grad_log(f, z_alto)[2]                       # usa 1,8
    errado = p.grad_log(f, np.array([z[0], z[1], 1.0]))[2]  # usa o recortado 1,0
    assert certo > errado, "o bruto tem de dar sinal MAIOR que o recortado"
    assert certo == pytest.approx((1.8 - 0.9) / 0.01)
    assert errado == pytest.approx((1.0 - 0.9) / 0.01)


def test_grad_log_e_a_pontuacao_gaussiana():
    """`∇_f log π = (z − f)/σ²`, em R³ — sem correção de fase."""
    p = NormalProjetada(sigma=0.5)
    f = np.array([1.0, 2.0, 0.3])
    z = np.array([1.5, 1.0, 0.9])
    assert np.allclose(p.grad_log(f, z), (z - f) / 0.25)


def test_concentracao_e_a_norma_das_duas_primeiras():
    """Diagnóstico obrigatório da §4.3: distingue "não aprendeu" de "colapsou
    para determinístico cedo demais" — dois problemas com correções opostas."""
    assert NormalProjetada.concentracao(np.array([3.0, 4.0, 0.5])) == pytest.approx(5.0)
    assert NormalProjetada.concentracao(np.zeros(3)) == 0.0


def test_avaliacao_e_deterministica():
    """Protocolo de Lamenza §7: *"during an evaluation the training is
    temporarily halted and action noise is removed"*. Não é escolha nossa."""
    p = NormalProjetada(sigma=0.5)
    f = np.array([1.0, 1.0, 0.6])
    a = p.media(f)
    assert a.theta_norm == pytest.approx(0.125)      # atan2(1,1) = 45°
    assert a.s_norm == pytest.approx(0.6)
    assert all(p.media(f) == a for _ in range(5)), "sem ruído, sempre igual"


def test_anelamento_de_sigma_e_linear_e_opcional():
    """Com `σ` fixa a política ótima determinística é inatingível e sobra ruído
    residual no treino."""
    fixa = NormalProjetada(sigma=0.2)
    for _ in range(100):
        fixa.contar_atualizacao()
    assert fixa.sigma_corrente == pytest.approx(0.2)

    anelada = NormalProjetada(sigma=0.2, sigma_min=0.02, decaimento=10)
    assert anelada.sigma_corrente == pytest.approx(0.2)
    for _ in range(5):
        anelada.contar_atualizacao()
    assert anelada.sigma_corrente == pytest.approx(0.11)
    for _ in range(100):
        anelada.contar_atualizacao()
    assert anelada.sigma_corrente == pytest.approx(0.02)


def test_sigma_invalida_falha_cedo():
    with pytest.raises(ValueError, match="sigma"):
        NormalProjetada(sigma=0.0)


def test_z_de_tamanho_errado_falha_cedo():
    with pytest.raises(ValueError, match="esperado 3"):
        NormalProjetada.para_acao(np.array([1.0, 0.0]))


# --- §4.8: a MLP em NumPy puro ------------------------------------------


def test_gradiente_da_mlp_bate_com_diferencas_finitas():
    """**O guarda principal.**

    Retropropagação à mão erra em silêncio, e o modo de falha é perverso: um
    baseline fraco *infla* H1. Aqui cada parâmetro é conferido individualmente.
    """
    rng = np.random.default_rng(0)
    m = MLP(entrada=7, saida=3, oculta=5, seed=1)
    x = rng.normal(size=(4, 7))
    alvo = rng.normal(size=(4, 3))

    def perda():
        return float(((m.prever(x) - alvo) ** 2).sum())

    g = m.gradientes(x, 2 * (m.prever(x) - alvo))
    h, pior = 1e-6, 0.0
    for p, gp in zip(m.parametros(), g):
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            orig = p[i]
            p[i] = orig + h; mais = perda()
            p[i] = orig - h; menos = perda()
            p[i] = orig
            num = (mais - menos) / (2 * h)
            pior = max(pior, abs(num - gp[i]) / max(1.0, abs(num)))
    assert pior < 1e-6, f"gradiente analítico divergiu {pior:.2e} do numérico"


def test_mlp_nao_importa_torch():
    """RNF-11: NumPy puro no caminho do experimento.

    Medir a latência de uma MLP `31→32→32→3` via PyTorch mede o **despacho** do
    PyTorch, não o modelo — e como a nossa WNN é NumPy, isso inverteria a
    conclusão de T3 por artefato de biblioteca.
    """
    import inspect

    from uavwnn.agents import mlp_ac

    fonte = inspect.getsource(mlp_ac)
    assert "import torch" not in fonte
    assert "torch." not in fonte.replace("PyTorch", "").replace("pytorch", "")


def test_relu_zera_o_gradiente_onde_a_ativacao_e_negativa():
    """Erro clássico: usar a pré-ativação em vez da ativação na máscara."""
    m = MLP(entrada=2, saida=1, oculta=3, seed=0)
    m.W[0][:] = np.array([[-10.0, -10.0, -10.0], [-10.0, -10.0, -10.0]])
    m.b[0][:] = -5.0                       # todas as ocultas mortas
    g = m.gradientes(np.array([[1.0, 1.0]]), np.array([[1.0]]))
    assert np.allclose(g[0], 0.0), "camada morta não deveria receber gradiente"


def test_adam_normaliza_e_converge():
    """Adam e não descida simples — a SPEC-03 mediu que descida simples deixava a
    MLP de clonagem em 23,3° contra 1,1°, o que teria "provado" a superioridade
    da WNN por subtreino."""
    alvo = np.array([3.0, -2.0])
    p = [np.zeros(2)]
    opt = Adam(taxa=0.1)
    for _ in range(500):
        opt.passo(p, [2 * (p[0] - alvo)])
    assert np.allclose(p[0], alvo, atol=1e-2)


def test_contagem_de_parametros_da_mlp_pareada():
    """A tabela de T1 conta pesos **e** vieses."""
    m = MLP(25, 3, 32)
    esperado = (25 * 32 + 32) + (32 * 32 + 32) + (32 * 3 + 3)
    assert m.n_parametros() == esperado
    assert m.bytes_fp32() == esperado * 4
    assert m.multiplicacoes_fp() == 25 * 32 + 32 * 32 + 32 * 3


def test_wnn_faz_zero_multiplicacoes_e_a_mlp_nao():
    """O contraste central de T3, agora com os dois lados no mesmo laço.

    A CHAVE MUDOU DE NOME, E A MUDANÇA É SUBSTANTIVA. Este teste checava
    `mult_fp_ator == 0`, e essa contagem **incluía o amostrador** — Box-Muller
    (`sqrt`, `log`, `sin`, `cos`) e o `atan2` da normal projetada, que fazem 4
    multiplicações em ponto flutuante e são **comuns aos dois braços**. O zero
    passava por acidente da implementação anterior.

    O PRD §5.5 já exige a separação: *"a alegação verificável é sobre
    multiplicações, e sobre o aproximador — o codificador e o amostrador entram
    na conta em linhas próprias"*. Então a comparação certa é entre
    `mult_fp_aproximador`, e o caminho completo é reportado ao lado, não escondido
    nem atribuído a um dos lados.
    """
    term = TermometroArtigo(-1.0, 1.0, 8)
    nt = NTuplaDissertacaoAprox(term.n_bits_para(25)).custo()
    mlp = MlpAprox(25).custo()

    assert nt["mult_fp_aproximador"] == 0          # a alegação de H1
    assert mlp["mult_fp_aproximador"] > 1000

    # O amostrador é o MESMO nos dois, e não é zero em nenhum.
    amostrador_nt = nt["mult_fp_caminho_completo"] - nt["mult_fp_aproximador"]
    amostrador_mlp = mlp["mult_fp_caminho_completo"] - mlp["mult_fp_aproximador"]
    assert amostrador_nt == amostrador_mlp > 0


# --- pareamento auditável — T-05.6 §3 ----------------------------------


def test_os_dois_aproximadores_satisfazem_a_interface_congelada():
    """Entram no harness da SPEC-02 sem caminho especial."""
    term = TermometroArtigo(-1.0, 1.0, 8)
    for ap, cod in ((MlpAprox(25), lambda v: v),
                    (NTuplaDissertacaoAprox(term.n_bits_para(25)),
                     lambda v: term.codificar(np.clip(v, -1, 0.999)[None, :])[0])):
        ag = AtorCriticoContinuo(aproximador=ap, codificar=cod)
        assert isinstance(ag, Policy)
        assert ag.usa_estado_verdadeiro is False
        assert "deterministica" in ag.metadados()


def test_pareamento_a_mesma_observacao_chega_aos_dois():
    """O que torna a comparação sobre o **aproximador** e não sobre a entrada.

    Espiona o que cada codificador recebe: tem de ser o mesmo vetor de
    observação, na mesma ordem.
    """
    vistos: dict[str, list] = {"mlp": [], "wnn": []}
    term = TermometroArtigo(-1.0, 1.0, 8)

    def cod_mlp(v):
        vistos["mlp"].append(np.array(v)); return np.asarray(v, float)

    def cod_wnn(v):
        vistos["wnn"].append(np.array(v))
        return term.codificar(np.clip(v, -1, 0.999)[None, :])[0]

    rng = np.random.default_rng(3)
    vetores = rng.uniform(-1, 1, size=(6, 25))
    for ap, cod in ((MlpAprox(25), cod_mlp),
                    (NTuplaDissertacaoAprox(term.n_bits_para(25)), cod_wnn)):
        ag = AtorCriticoContinuo(aproximador=ap, codificar=cod)
        ag.reset(_ctx(), np.random.default_rng(0))
        for v in vetores:
            ag.act(_obs(v))
    assert len(vistos["mlp"]) == len(vistos["wnn"]) == 6
    for a, b in zip(vistos["mlp"], vistos["wnn"]):
        assert np.array_equal(a, b), "os dois têm de ver a MESMA observação"


def test_avaliacao_nao_atualiza_nada():
    """`treinando = False` é modo de avaliação: média, sem ruído, sem update."""
    ap = MlpAprox(25)
    antes = [p.copy() for p in ap.ator.parametros()]
    ag = AtorCriticoContinuo(aproximador=ap, codificar=lambda v: v,
                             treinando=False)
    ag.reset(_ctx(), np.random.default_rng(0))
    for _ in range(5):
        ag.act(_obs(np.zeros(25)))
    ag.fim_de_episodio()
    for a, b in zip(antes, ap.ator.parametros()):
        assert np.array_equal(a, b), "avaliação não pode treinar"


# --- os dois aprendem — T-05.4 -----------------------------------------


def _bandido(aprox, cod, taxa_msg, passos=6000, seed=1):
    """Bandido contextual contínuo: recompensa `−Δθ² − 3(s−s*)²`.

    Exercita **as duas** componentes da ação, e o alvo de rumo inclui valores
    perto de `2π` de propósito — é onde o wrap-around apareceria se estivesse
    errado.
    """
    D, NC = 6, 4
    rng = np.random.default_rng(7)
    cv = rng.normal(size=(NC, D))
    th_a = np.array([0.3, 2.0, 3.8, 5.4])
    s_a = np.array([0.9, 0.4, 0.7, 0.2])

    ag = AtorCriticoContinuo(
        aproximador=aprox, codificar=cod, passos=1, desconto=1.0,
        politica=NormalProjetada(0.25, 0.05, passos), treinando=True)
    ag.reset(_ctx(), np.random.default_rng(seed))
    for p in range(passos):
        i = p % NC
        a = ag.act(_obs(cv[i]))
        th = a.theta_norm * 2 * np.pi
        d = abs(np.mod(th - th_a[i] + np.pi, 2 * np.pi) - np.pi)
        ag.registrar(-(d ** 2) - 3.0 * (a.s_norm - s_a[i]) ** 2)
        ag.fim_de_episodio()

    ag.treinando = False
    et, es = [], []
    for i in range(NC):
        a = ag.act(_obs(cv[i]))
        th = a.theta_norm * 2 * np.pi
        et.append(abs(np.mod(th - th_a[i] + np.pi, 2 * np.pi) - np.pi))
        es.append(abs(a.s_norm - s_a[i]))
    return float(np.degrees(np.mean(et))), float(np.mean(es))


def test_mlp_aprende_o_bandido():
    """Se a MLP não resolve o brinquedo, o gradiente está errado — não é o
    problema que é difícil.

    `taxa = 1e-4` é o ótimo medido numa varredura de 3e-5 a 3e-2, e a curva é
    **não-monótona com ótimo agudo**: 15,6° em 3e-5, 0,8° em 1e-4, 2,8° em 1e-3,
    53° em 1e-2. Um palpite de um ponto só erraria — o meu primeiro chute foi
    30× alto, o que teria feito a MLP parecer 5× pior que a WNN.
    """
    et, es = _bandido(MlpAprox(6, taxa_ator=1e-4, taxa_critico=1e-4, seed=0),
                      lambda v: np.asarray(v, float) / 3.0, "1e-4")
    assert et < 15.0, f"erro de rumo {et:.1f}° alto demais"
    assert es < 0.10, f"erro de velocidade {es:.3f} alto demais"


def test_ntuple_aprende_o_bandido():
    term = TermometroArtigo(-3.0, 3.0, 8)
    et, es = _bandido(
        NTuplaDissertacaoAprox(term.n_bits_para(6), tamanho_tupla=8, taxa_ator=3e-3, seed=0),
        lambda v: term.codificar(np.clip(v, -3, 2.999)[None, :])[0], "3e-3")
    assert et < 15.0, f"erro de rumo {et:.1f}° alto demais"
    assert es < 0.10, f"erro de velocidade {es:.3f} alto demais"


def test_sinal_do_gradiente_da_mlp_esta_certo():
    """Adam **desce**; a ascensão precisa do negativo.

    Com o sinal trocado a política foge do que funciona, e o erro cresceria em
    vez de cair. Aqui comparamos início e fim em vez de um limiar absoluto.
    """
    ap = MlpAprox(6, taxa_ator=1e-4, taxa_critico=1e-4, seed=0)
    cod = lambda v: np.asarray(v, float) / 3.0
    inicial, _ = _bandido(ap, cod, "", passos=1)
    ap2 = MlpAprox(6, taxa_ator=1e-4, taxa_critico=1e-4, seed=0)
    final, _ = _bandido(ap2, cod, "", passos=6000)
    assert final < inicial, f"{inicial:.1f}° -> {final:.1f}°: não melhorou"


def test_concentracao_media_e_acumulada_sem_guardar_cada_decisao():
    """A média dos metadados sai de soma e contagem. A versão anterior guardava
    uma lista com um float por `act` e nunca a esvaziava: 38 bytes por decisão,
    ~15 GB numa corrida de 50 M passos com 8 drones — as duas de `Na = 8`
    estavam a 3 GB em 10 M com LUTs de poucos MB."""
    from uavwnn.agents.ac_continuo import AtorCriticoContinuo

    class _Ap:
        def f(self, x):
            return np.asarray(x, float)

        def custo(self):
            return {}

    pol = AtorCriticoContinuo(aproximador=_Ap(), codificar=lambda v: v, treinando=False)
    assert np.isnan(pol.metadados()["concentracao_media"])
    for v in ([3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [6.0, 8.0, 0.0]):
        pol.act_vetor(np.array(v))
    assert pol.metadados()["concentracao_media"] == pytest.approx((5.0 + 0.0 + 10.0) / 3)
    assert not hasattr(pol, "_concentracoes")
    assert pol._conc_n == 3
