"""
Módulo externo para cálculo do supremo em malha com interface "Dürr–Høyer-ready".

Exporta:
- sup_abs_d_dh(alpha, beta, *, eps_q, m, backend) -> float
    Retorna max_{k=0..m} |alpha*(k/m) + beta - e^{k/m}| + eps_q,
    delegando o argmax a um 'backend' que implementa:
        backend.max_on_grid(f_callable, m) -> (i_max, f_max)

Inclui também um backend de referência simples (ClassicalGridBackend) para testes locais.
Substitua por um backend quântico (ex.: Qiskit/Braket) com a mesma API quando quiser.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol, Tuple


class MaxOnGridBackend(Protocol):
    def max_on_grid(self, f: Callable[[int], float], m: int) -> Tuple[int, float]:
        """Deve retornar (i_max, f_max) para i em {0,..,m}."""
        ...


def sup_abs_d_dh(
    alpha: float,
    beta: float,
    *,
    eps_q: float,
    m: int,
    backend: MaxOnGridBackend,
) -> float:
    """Calcula \hat S(α,β;ε_q) via máximo em malha + colchão ε_q, delegando o argmax ao backend.

    Parâmetros
    -----------
    alpha, beta : coeficientes da aproximação afim α t + β
    eps_q       : colchão de segurança (discretização + ruído/estocástico)
    m           : número de subintervalos; malha T_m = {0, 1/m, ..., 1}
    backend     : objeto com método .max_on_grid(f, m) -> (i_max, f_max)

    Retorna
    --------
    float : max_{k=0..m} |α*(k/m) + β − e^{k/m}| + ε_q
    """
    if backend is None:
        raise RuntimeError(
            "Nenhum backend fornecido. Passe um objeto com .max_on_grid(f, m)->(i_max,f_max)."
        )
    if m <= 0:
        raise ValueError("m deve ser positivo (ex.: 1024)")
    if eps_q < 0:
        raise ValueError("eps_q deve ser não-negativo")

    h = 1.0 / float(m)

    def f(k: int) -> float:
        if k < 0 or k > m:
            # Fora do domínio; pode acontecer em implementações com N=2^n>m+1
            return float("-inf")
        t = k * h
        return abs(alpha * t + beta - math.exp(t))

    i_max, f_max = backend.max_on_grid(f, m)
    if not (isinstance(i_max, int) and 0 <= i_max <= m):
        raise RuntimeError(f"Backend retornou índice inválido: {i_max}")
    if not (isinstance(f_max, (int, float)) and f_max >= 0.0):
        raise RuntimeError(f"Backend retornou valor inválido: {f_max}")

    return float(f_max) + float(eps_q)


# -------------------- Backend de referência (clássico) --------------------

@dataclass
class ClassicalGridBackend:
    """Backend de referência que calcula o máximo em CPU (sem aceleração quântica).

    Útil para testar a integração fim-a-fim antes de plugar Grover/Dürr–Høyer.
    """

    def max_on_grid(self, f: Callable[[int], float], m: int) -> Tuple[int, float]:
        best_i = 0
        best_val = f(0)
        for k in range(1, m + 1):
            v = f(k)
            if v > best_val:
                best_i, best_val = k, v
        return best_i, float(best_val)


# -------------------- (Opcional) esqueleto para backend Qiskit --------------------
# class QiskitAerMaxFinderBackend:
#     def __init__(self, ...):
#         ...  # inicialize simulador / seed
#
#     def max_on_grid(self, f: Callable[[int], float], m: int) -> Tuple[int, float]:
#         # TODO: implementar Grover/Dürr–Høyer simulado no Aer
#         # - construir oráculo de marcação por limiar
#         # - laço de iterações aleatórias (técnica DH)
#         # - medir e atualizar threshold
#         raise NotImplementedError
