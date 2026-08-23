"""Invariante de vazamento RF-04 — SPEC-07 T-07.8.

A camada 4 é a prova: grava-e-repete o fluxo de ruído, desloca a verdade por `δ`
só no caminho de telemetria, e exige hash idêntico da sequência de ações. As
outras camadas (custódia por token, tipos distintos, marca em tempo de execução)
existem para que ela quase nunca falhe, e cada uma é testada no seu limite.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from uavwnn.fusion.kalman import FiltroKalmanControle
from uavwnn.fusion.ruido import CanalGPS, ConfigRuido, fluxo_ruido
from uavwnn.fusion.bancada import trajetoria_b1
from uavwnn.fusion.verdade import (
    ArrayMarcado,
    GravadorRuido,
    RegistroVerdade,
    ReprodutorRuido,
    TokenMetricas,
    esta_marcado,
)


# --- camada 4: o teste diferencial, a prova -------------------------------


def _politica(x_hat: np.ndarray) -> np.ndarray:
    """Uma política qualquer que só enxerga a estimativa (derivada da medida).

    Rumo para a origem, velocidade proporcional. O ponto não é ser boa; é
    depender EXCLUSIVAMENTE de `x_hat` — nunca da verdade.
    """
    ang = np.arctan2(-x_hat[1], -x_hat[0])
    s = min(1.0, np.linalg.norm(x_hat) / 100.0)
    return np.array([ang, s])


def _rodar(traj, canal, registro, deslocamento_verdade):
    """Roda um episódio, devolve o hash da sequência de ações.

    A verdade (deslocada por `δ`) vai SÓ para o `registro` (métricas). A política
    lê só a estimativa do filtro, que vem das medidas. Se algo vazar a verdade
    para a política, mudar `δ` muda as ações — e o hash quebra.
    """
    f = FiltroKalmanControle(sigma_gps=10.0, q_w=0.1, sigma_v0=0.1)
    t0 = float(traj.instantes[0])
    z0 = canal.medir(traj.posicao_em(t0), t0)
    f.iniciar(z0, t0)
    acoes = []
    for k in range(traj.n_passos):
        t = float(traj.instantes[k + 1])
        p_true = traj.posicao_em(t)
        z = canal.medir(p_true, t)
        f.comando(traj.comandos[k])
        f.atualizar(z, t)
        x_hat, _ = f.estimativa(t)
        acoes.append(_politica(x_hat))
        # a verdade, deslocada, entra só nas métricas
        registro.registrar(t, 0, p_true + deslocamento_verdade)
    arr = np.array(acoes, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def test_deslocar_a_verdade_nao_muda_as_acoes():
    """A prova de RF-04, em ≥ 20 pares (semente, δ)."""
    for par in range(20):
        rng = np.random.default_rng(par)
        traj = trajetoria_b1(rng, n_passos=60, sigma_salto=np.pi)

        # corrida A: canal real, δ = 0, grava o ruído
        canal_a = GravadorRuido(CanalGPS(ConfigRuido(sigma_gps=10.0), fluxo_ruido(1, "gps", 0, par)))
        hash_a = _rodar(traj, canal_a, RegistroVerdade(), np.zeros(2))

        # corrida B: reproduz o ruído gravado, desloca a verdade por δ
        delta = rng.normal(scale=50.0, size=2)
        canal_b = ReprodutorRuido(canal_a.gravado)
        hash_b = _rodar(traj, canal_b, RegistroVerdade(), delta)

        assert hash_a == hash_b, f"par {par}: a verdade vazou para a política"


def test_o_reprodutor_realmente_ignora_a_verdade():
    """Sanidade do arnês: o reprodutor devolve o gravado, não a verdade recebida."""
    grav = GravadorRuido(CanalGPS(ConfigRuido(sigma_gps=3.0), fluxo_ruido(1, "gps", 0, 0)))
    zs = [grav.medir(np.array([float(k), 0.0]), float(k)) for k in range(5)]
    rep = ReprodutorRuido(grav.gravado)
    for k in range(5):
        # passa uma verdade completamente diferente; deve ignorar
        z = rep.medir(np.array([9999.0, 9999.0]), float(k))
        assert np.allclose(z, zs[k])


# --- camada 1: custódia por token -----------------------------------------


def test_token_nao_pode_ser_construido_fora_dos_modulos_autorizados():
    with pytest.raises(PermissionError):
        TokenMetricas()  # este módulo (tests) não está na allowlist


def test_ler_a_verdade_exige_token():
    reg = RegistroVerdade()
    reg.registrar(1.0, 0, np.array([1.0, 2.0]))
    with pytest.raises(PermissionError):
        reg.ler(object(), 1.0, 0)  # type: ignore[arg-type]


def test_registrar_nao_devolve_a_verdade():
    """A camada 1 escreve sem retorno — não há como ler por `registrar`."""
    reg = RegistroVerdade()
    assert reg.registrar(1.0, 0, np.array([1.0, 2.0])) is None


# --- camada 3: marca em tempo de execução ----------------------------------


def test_marca_propaga_por_operacoes():
    """`p_true − x̂` chega marcado à observação (o vazamento por valor derivado)."""
    verdade = ArrayMarcado(np.array([10.0, 20.0]))
    estimativa = np.array([1.0, 2.0])
    assert esta_marcado(verdade)
    assert esta_marcado(verdade - estimativa), "diferença herda a marca"
    assert esta_marcado(verdade[:1]), "fatia herda a marca"
    assert esta_marcado(verdade * 2), "escala herda a marca"
    assert not esta_marcado(estimativa), "array limpo não é marcado"


def test_a_marca_nao_sobrevive_a_escalar_limite_documentado():
    """Limite conhecido: a marca some em `float()`/`.item()` — por isso a camada 4
    é a prova, não esta."""
    verdade = ArrayMarcado(np.array([10.0, 20.0]))
    assert not esta_marcado(float(verdade[0]))
    assert not esta_marcado(verdade[0].item())


# --- RNF-07: fusion é separável -------------------------------------------


def test_refenv_nao_importa_fusion():
    """Com `fusion/` desligado, o sistema é o Lamenza puro (RNF-07).

    O ambiente de referência não pode depender do pacote de extensão — é o que
    torna o princípio da §0 do PRD verificável por construção.
    """
    import ast
    from pathlib import Path

    raiz = Path(__file__).resolve().parents[1] / "src" / "uavwnn" / "refenv"
    for arq in raiz.glob("*.py"):
        arvore = ast.parse(arq.read_text())
        for no in ast.walk(arvore):
            if isinstance(no, ast.ImportFrom) and no.module and "fusion" in no.module:
                raise AssertionError(f"{arq.name} importa fusion")
            if isinstance(no, ast.Import):
                for alias in no.names:
                    assert "fusion" not in alias.name, f"{arq.name} importa {alias.name}"
