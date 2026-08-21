"""Critério de saída 3 da SPEC-05 — **o pareamento, auditável por máquina**.

> Um teste automatizado que instancia `NTupleAC` e `MlpAC` com a mesma semente e
> verifica que a sequência de `(obs, reward, terminal/truncated)` consumida e a
> forma da ação emitida são idênticas passo a passo. **O teste falha se qualquer
> assimetria for introduzida.**

O QUE ESTE TESTE PODE E NÃO PODE AFIRMAR

Ele **não** compara trajetórias no ambiente. Não pode: as duas políticas emitem
ações diferentes, então a partir do segundo passo elas veriam estados diferentes,
e exigir igualdade ali seria exigir que os dois aproximadores fossem o mesmo — o
oposto do que a comparação quer.

O que ele verifica é o **encanamento**: dirigidos pela MESMA fita de
`(observação, recompensa, terminal)`, os dois consomem exatamente as mesmas
entradas, na mesma ordem, e fazem o mesmo NÚMERO de atualizações no mesmo
INSTANTE. É essa simetria que a comparação de H1 pressupõe, e é ela que uma
refatoração distraída quebra — um braço ganhando uma atualização extra, ou vendo
a recompensa deslocada em um passo, é o tipo de viés que não aparece em nenhuma
métrica agregada e inverte a conclusão.

A fita torna o teste determinístico e independente do simulador, que é o que
permite rodá-lo em cada commit.
"""

from __future__ import annotations

import numpy as np
import pytest

from uavwnn.agents.ac_continuo import AtorCriticoContinuo, MlpAprox, NTuplaAprox
from uavwnn.agents.base import Acao, ContextoEpisodio, Observacao
from uavwnn.agents.politica_continua import D_ATOR, NormalProjetada

CTX = ContextoEpisodio(num_drones=2, num_sensores=8, scenario_size=200.0,
                       communication_range=20.0, dt=2.0)
N_PASSOS = 24
SEMENTE = 20260821


def fita(n: int = N_PASSOS) -> list[tuple[np.ndarray, float, bool]]:
    """`n` triplas `(vetor de observação, recompensa, terminal)`, fixas.

    Gerada com semente própria e devolvida como lista concreta — não como
    gerador — para que os dois braços consumam **o mesmo objeto**, e não duas
    realizações de um mesmo processo aleatório.
    """
    g = np.random.default_rng(SEMENTE)
    d = 2 * (CTX.num_drones - 1) + 2 * CTX.state_num_closest_sensors + 1
    saida = []
    for t in range(n):
        v = g.uniform(0.3, 0.7, size=d)
        v[-1] = 0.0                                  # índice do agente
        saida.append((v, float(g.normal()), t == n - 1))
    return saida


class Espiao:
    """Envolve um aproximador e registra tudo que passa por ele.

    Não altera nada: delega cada chamada e anota. O registro é o que o teste
    compara entre os dois braços.
    """

    def __init__(self, alvo) -> None:
        self._alvo = alvo
        self.registro: list[tuple] = []

    def f(self, x):
        self.registro.append(("f", np.asarray(x).copy()))
        return self._alvo.f(x)

    def v(self, x):
        self.registro.append(("v", np.asarray(x).copy()))
        return self._alvo.v(x)

    def ascender_ator(self, x, direcao):
        self.registro.append(("ascender", np.asarray(x).copy(),
                              np.asarray(direcao).shape))
        return self._alvo.ascender_ator(x, direcao)

    def regredir_critico(self, x, alvo):
        self.registro.append(("regredir", np.asarray(x).copy()))
        return self._alvo.regredir_critico(x, alvo)

    def custo(self):
        return self._alvo.custo()


BITS_POR_ESCALAR = 4
D_OBS = 2 * (CTX.num_drones - 1) + 2 * CTX.state_num_closest_sensors + 1
N_BITS = D_OBS * BITS_POR_ESCALAR
"""Derivado da construção, e não fixado à mão: um limiar por faixa, `D_OBS`
escalares. Fixar `n_bits` independentemente já causou um `IndexError` — o
mapeamento particionado exige que os dois números fechem."""


def _codificador():
    """Codificação de brinquedo, mas com a assinatura real.

    A n-tupla precisa de bits e a MLP de floats, então os codificadores são
    necessariamente diferentes — é justamente a assimetria **permitida** pela
    §4.8 ("os dois recebem a mesma observação e cada um a prepara à sua
    maneira"). O que o teste garante é que ela é a **única**.
    """
    lim = np.linspace(0.3, 0.7, BITS_POR_ESCALAR + 2)[1:-1]

    def cod(vetor):
        v = np.asarray(vetor, float).ravel()
        return (v[:, None] > lim[None, :]).ravel()
    return cod


def _rodar(aprox, codificar) -> tuple[list, list[Acao]]:
    """Dirige uma política pela fita e devolve `(registro do espião, ações)`."""
    espiao = Espiao(aprox)
    pol = AtorCriticoContinuo(
        aproximador=espiao, codificar=codificar,
        politica=NormalProjetada(0.1, None), passos=8, treinando=True)
    pol.reset(CTX, np.random.default_rng(SEMENTE))
    acoes = []
    for vetor, recompensa, terminal in fita():
        obs = Observacao.decodificar(vetor, CTX)
        acoes.append(pol.act(obs))
        pol.registrar(recompensa)
        if terminal:
            pol.fim_de_episodio()
        else:
            pol.passo_intermediario(obs)
    return espiao.registro, acoes


@pytest.fixture(scope="module")
def dois_bracos():
    ntupla = _rodar(NTuplaAprox(n_bits=N_BITS, tamanho_tupla=8, seed=0),
                    _codificador())
    mlp = _rodar(MlpAprox(n_bits_ou_dim=D_OBS, seed=0),
                 lambda v: np.asarray(v, float))
    return ntupla, mlp


def test_mesma_sequencia_de_chamadas(dois_bracos):
    """A ordem e o tipo das operações são idênticos — o coração do critério 3."""
    (reg_nt, _), (reg_mlp, _) = dois_bracos
    tipos_nt = [r[0] for r in reg_nt]
    tipos_mlp = [r[0] for r in reg_mlp]
    assert tipos_nt == tipos_mlp, (
        "os dois braços fizeram sequências de operações diferentes; alguma "
        "assimetria entrou no laço do Algoritmo 4.4")


def test_mesmo_numero_de_atualizacoes(dois_bracos):
    """Um braço com uma atualização a mais vê mais dados pelo mesmo orçamento."""
    (reg_nt, _), (reg_mlp, _) = dois_bracos
    for op in ("ascender", "regredir", "f", "v"):
        n_nt = sum(1 for r in reg_nt if r[0] == op)
        n_mlp = sum(1 for r in reg_mlp if r[0] == op)
        assert n_nt == n_mlp, f"'{op}': n-tupla {n_nt} vezes, MLP {n_mlp} vezes"


def test_toda_atualizacao_do_ator_tem_a_forma_da_politica(dois_bracos):
    """A direção subida é de dimensão `d = 3` — a normal projetada, nos dois."""
    for reg, _ in dois_bracos:
        formas = {r[2] for r in reg if r[0] == "ascender"}
        assert formas == {(D_ATOR,)}, formas


def test_acoes_no_dominio_valido(dois_bracos):
    """`Acao.__post_init__` já rejeita fora de `[0,1]`; aqui o par e a contagem."""
    for _, acoes in dois_bracos:
        assert len(acoes) == N_PASSOS
        for a in acoes:
            assert 0.0 <= a.theta_norm <= 1.0 and 0.0 <= a.s_norm <= 1.0


def test_o_ator_ve_o_estado_da_transicao_nao_o_seguinte(dois_bracos):
    """Desalinhamento de um passo entre `x` e a recompensa é o bug clássico.

    A entrada de cada `ascender` tem de ser idêntica à do `f`/`v` que a
    precederam para aquela transição — se o laço guardasse `S_{t+1}` no lugar de
    `S_t`, a atualização creditaria a recompensa ao estado errado. Isso não
    aparece em nenhuma média e degrada os dois braços de forma desigual.
    """
    for reg, _ in dois_bracos:
        for i, r in enumerate(reg):
            if r[0] != "ascender":
                continue
            anterior = [q for q in reg[:i] if q[0] in ("f", "v")]
            assert anterior, "ascender sem f/v antes"
            np.testing.assert_allclose(r[1], anterior[-1][1], err_msg=(
                "a atualização do ator usou uma entrada diferente da que o "
                "crítico avaliou — desalinhamento de passo"))


def test_avaliacao_nao_atualiza_nem_amostra():
    """Com `treinando=False` a ação é a média e nada é escrito (Lamenza §7)."""
    aprox = MlpAprox(n_bits_ou_dim=D_OBS, seed=0)
    espiao = Espiao(aprox)
    pol = AtorCriticoContinuo(aproximador=espiao, codificar=lambda v: np.asarray(v, float),
                              politica=NormalProjetada(0.1, None), treinando=False)
    pol.reset(CTX, np.random.default_rng(0))
    for vetor, _, _ in fita(6):
        pol.act(Observacao.decodificar(vetor, CTX))
    assert not [r for r in espiao.registro if r[0] in ("ascender", "regredir")]

    # determinística: duas passagens dão a MESMA ação (crit. de saída 7)
    obs = Observacao.decodificar(fita(1)[0][0], CTX)
    a, b = pol.act(obs), pol.act(obs)
    assert (a.theta_norm, a.s_norm) == (b.theta_norm, b.s_norm)


def test_determinismo_bit_a_bit_da_tabela_n_tupla():
    """Critério de saída 7: mesma semente, tabelas idênticas bit a bit."""
    def treinar():
        aprox = NTuplaAprox(n_bits=N_BITS, tamanho_tupla=8, seed=0)
        _rodar(aprox, _codificador())
        return aprox

    a, b = treinar(), treinar()
    for rede_a, rede_b in ((a.ator, b.ator), (a.critico, b.critico)):
        assert len(rede_a.memoria) == len(rede_b.memoria)
        for ma, mb in zip(rede_a.memoria, rede_b.memoria):
            assert ma.keys() == mb.keys()
            for chave in ma:
                va, vb = ma[chave], mb[chave]
                if isinstance(va, tuple):          # crítico: (valor, contador)
                    for xa, xb in zip(va, vb):
                        assert np.array_equal(xa, xb)
                else:
                    assert np.array_equal(va, vb)


def test_src_nao_importa_torch():
    """Critério de saída 5, na leitura que ele realmente protege.

    A redação da spec diz "nenhuma dependência de PyTorch fora de `tests/`", mas
    os scripts `spec02_maddpg_*.py` importam `torch` por **necessidade**: é o que
    lê os pesos publicados de Lamenza, e o baseline MADDPG não existe sem eles. O
    que a regra protege é a nossa implementação — `src/uavwnn/` — que precisa
    rodar sem framework (RNF-11). É isso que este teste verifica.
    """
    import pathlib
    raiz = pathlib.Path(__file__).resolve().parents[1] / "src" / "uavwnn"
    culpados = [p.relative_to(raiz) for p in raiz.rglob("*.py")
                if "torch" in p.read_text()]
    assert not culpados, f"torch citado em src/uavwnn: {culpados}"


# ---------------------------------------------------------------------------
# critério de saída 4 — as faixas registradas, com procedência
# ---------------------------------------------------------------------------


def _config():
    import pathlib
    import tomllib
    caminho = (pathlib.Path(__file__).resolve().parents[1]
               / "configs" / "spec05" / "hiperparametros.toml")
    return tomllib.loads(caminho.read_text())


def test_toda_faixa_tem_procedencia():
    """Uma faixa sem origem declarada é uma escolha não justificada.

    É o que o critério 4 impede: o arquivo existe para que nenhum número entre na
    varredura da SPEC-06 sem dizer se vem do problema (Lamenza), do método
    (Katopodis) ou de medição nossa.
    """
    cfg = _config()
    assert cfg, "configuração vazia"
    for nome, entrada in cfg.items():
        assert entrada.get("origem") in ("lamenza", "katopodis", "propria"), nome
        assert entrada.get("nota", "").strip(), f"{nome} sem justificativa"


def test_gamma_fixo_em_099():
    """`γ = 0,99` é de Lamenza e não entra em varredura (critério 4)."""
    gamma = _config()["gamma"]
    assert gamma["valor"] == 0.99
    assert gamma["fixo"] is True
    assert gamma["origem"] == "lamenza"


def test_config_concorda_com_o_codigo():
    """O arquivo tem de descrever o que o código faz, ou é ficção.

    Verifica os três acoplamentos que já se desalinharam uma vez: `d = 3` da
    normal projetada, `c_max = 0,5` do ator, e `c_max` nulo no crítico.
    """
    from uavwnn.wnn.redes import RedeRegressaoEsquecimento
    cfg = _config()
    assert cfg["d_ator"]["valor"] == D_ATOR
    # `c_max` é lido de `NTuplaAprox`, e não da rede: o padrão vive no ponto de
    # montagem do agente, não na primitiva. Ver
    # `test_asseracao_de_estabilidade_ligada_no_caminho_de_treino`.
    assert NTuplaAprox(n_bits=N_BITS, tamanho_tupla=8).ator.c_max == cfg["c_max"]["valor"]
    assert RedeRegressaoEsquecimento(32, 8).c_max is None
    assert cfg["c_max"]["valor_critico"] == "nulo"
    assert str(D_ATOR) in cfg["arquitetura_mlp"]["valor"]


def test_asseracao_de_estabilidade_ligada_no_caminho_de_treino():
    """Critério de saída 6 — e o teste existe porque o padrão mudou de lugar.

    A asserção da §4.2 é `None` na rede (primitiva de uso geral, que serve de
    oráculo contra o `Ramnet.jl`) e `0,5` em `NTuplaAprox` (onde o gradiente de
    política entra, e onde a divergência da RNF-01 foi medida). Se alguém inverter
    esse padrão, a proteção desaparece sem nenhum teste ficar vermelho — salvo
    este.
    """
    from uavwnn.wnn.redes import TreinoInstavel
    aprox = NTuplaAprox(n_bits=N_BITS, tamanho_tupla=8, seed=0)
    assert aprox.ator.c_max == 0.5, "asserção desligada no caminho de treino"

    bits = np.zeros(N_BITS, bool)
    bits[:8] = True
    aprox.ascender_ator(bits, np.full(D_ATOR, 0.01))            # dentro do teto
    with pytest.raises(TreinoInstavel, match="excede c_max"):
        aprox.ascender_ator(bits, np.full(D_ATOR, 500.0))
    with pytest.raises(TreinoInstavel, match="não finito"):
        aprox.ascender_ator(bits, np.full(D_ATOR, np.nan))


def test_desligar_a_asseracao_e_decisao_explicita():
    """`c_max=None` funciona, mas tem de ser escrito por quem monta o agente."""
    aprox = NTuplaAprox(n_bits=N_BITS, tamanho_tupla=8, seed=0, c_max=None)
    bits = np.zeros(N_BITS, bool)
    bits[:8] = True
    aprox.ascender_ator(bits, np.full(D_ATOR, 500.0))           # passa
    assert aprox.ator.c_max is None
