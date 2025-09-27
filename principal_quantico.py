"""
Implementação do algoritmo RCP-H&R-PLSI (versão clássica com isotropização e corte pelo objetivo)
para o problema de melhor aproximação uniforme de e^t em [0,1] por funções afins α t + β.

Nesta versão, removemos a compactação artificial 0<=gamma<=Gamma. O conjunto de trabalho por iteração é definido apenas pelas restrições originais da PLSI (sup_abs_d(alpha,beta) - gamma <= 0) e pelo corte objetivo gamma <= tau_k, que é atualizado a cada iteração com a segunda melhor amostra. Isso torna o método fiel ao pseudocódigo clássico: o encolhimento do corpo vem exclusivamente do corte pelo objetivo.

Requisitos: numpy.

Erro de discretização: para r(t) = αt + β − e^t tem-se |r'(t)| = |α − e^t| ≤ |α| + e. Logo

| sup_{[0,1]}|r| − max_{t∈T_m}|r| | ≤ h max_{t∈[0,1]}|r'(t)| ≤ h (|α| + e).
"""
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable

import numpy as np
from qiskit_backend_dh import QiskitGroverSetBackend
from dh_oracle import ClassicalGridBackend as _ClassicalGridBackend  # <— import fixo no topo

# Solver externo do supremo (Dürr–Høyer ou simulado)
try:
    from dh_oracle import sup_abs_d_dh  # assinatura: (alpha, beta, *, eps_q: float, m: int, backend) -> float
except Exception as e:
    sup_abs_d_dh = None
    _import_error_dh = e

# ============================================================
# Núcleo do problema: violação usando supremo externo (Dürr–Høyer)
# ============================================================

def _sup_on_grid_quantum(alpha: float, beta: float, eps_q: float, m: int, backend) -> float:
    """Wrapper fino para chamar o solver externo de supremo."""
    if sup_abs_d_dh is None:
        raise RuntimeError(
            f"Módulo externo dh_oracle não encontrado ou inválido: {_import_error_dh}"
        )
    return sup_abs_d_dh(alpha, beta, eps_q=eps_q, m=m, backend=backend)


def make_phi_quantum(params: "Params"):
    """Constrói φ(x)=S_hat(α,β;ε_q)−γ usando apenas o solver externo.
    Se adaptive_grid=True, escolhe m para garantir h(|α|+e)≤ε_q.
    """
    def phi(x: np.ndarray) -> float:
        alpha, beta, gamma = float(x[0]), float(x[1]), float(x[2])
        if params.adaptive_grid:
            m = max(1, int(np.ceil((abs(alpha) + math.e) / max(1e-16, params.eps_q))))
        else:
            m = params.m_grid
        S = _sup_on_grid_quantum(alpha, beta, params.eps_q, m, params.quantum_backend)
        return S - gamma
    return phi


def sup_external(alpha: float, beta: float, params: "Params") -> float:
    """Convenience: calcula S_hat(α,β;ε_q) usando mesmas regras de malha de make_phi_quantum."""
    if params.adaptive_grid:
        m = max(1, int(np.ceil((abs(alpha) + math.e) / max(1e-16, params.eps_q))))
    else:
        m = params.m_grid
    return _sup_on_grid_quantum(alpha, beta, params.eps_q, m, params.quantum_backend)

# ============================================================
# Bracketing e oráculo de fronteira sem compactação (usa apenas phi<=0 e gamma<=tau)
# ============================================================

def bisect_zero(psi: Callable[[float], float], left: float, right: float, tol: float) -> float:
    """Bisseção *robusta* para resolver psi(λ)=0."""
    l, r = float(left), float(right)
    if l > r:
        l, r = r, l
    pl, pr = psi(l), psi(r)

    eps_psi = 1e-12
    if (pl <= eps_psi and pr >= -eps_psi) or (pl <= 0.0 and pr >= 0.0):
        pass
    else:
        if pl > 0.0 and pr <= 0.0:
            l, r = r, l
            pl, pr = pr, pl
        if pl <= 0.0 and pr <= 0.0:
            return r
        if pl > 0.0 and pr > 0.0:
            for _ in range(80):
                m = 0.5 * (l + r)
                pm = psi(m)
                if pm <= 0.0:
                    l, pl = m, pm
                    break
                else:
                    r, pr = m, pm
            if pl > 0.0 and pr > 0.0:
                return l if pl < pr else r

    while abs(r - l) > tol:
        m = 0.5 * (l + r)
        pm = psi(m)
        if pm <= 0.0:
            l = m
        else:
            r = m
    return 0.5 * (l + r)


def halfspace_interval_for_tau(y: np.ndarray, v: np.ndarray, tau: float) -> Tuple[float, float]:
    """Intervalo de lambda tal que (y3 + lambda*v3) <= tau."""
    y3 = float(y[2])
    v3 = float(v[2])
    if v3 > 0.0:
        lam_right = (tau - y3) / v3
        lam_left = -math.inf
    elif v3 < 0.0:
        lam_left = (tau - y3) / v3
        lam_right = +math.inf
    else:
        if y3 <= tau:
            lam_left, lam_right = -math.inf, +math.inf
        else:
            raise ValueError("Ponto y não atende gamma<=tau")
    return float(lam_left), float(lam_right)


def boundary_oracle(
    y: np.ndarray,
    v: np.ndarray,
    tau: float,
    phi: Callable[[np.ndarray], float],
    *,
    eps_lambda: float = 1e-9,
    init_step: float = 1.0,
    max_expand: int = 64,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Segmento factível de y+lambda v considerando apenas phi(x)<=0 e gamma<=tau."""
    def psi(lmbd: float) -> float:
        return phi(y + lmbd * v)


    lam_left_h, lam_right_h = halfspace_interval_for_tau(y, v, tau)

    # Direita
    lam_prev = 0.0
    lam_last_feas_right = 0.0
    lam = min(init_step, lam_right_h)
    k = 0
    while True:
        val = psi(lam)
        if val <= 0.0:
            lam_last_feas_right = lam
        if lam >= lam_right_h - 0.0:
            if val <= 0.0:
                lam_plus = lam
                x_plus = y + lam_plus * v
                break
            else:
                lam_plus = bisect_zero(psi, lam_prev, lam, eps_lambda)
                x_plus = y + lam_plus * v
                break
        if val > 0.0:
            lam_plus = bisect_zero(psi, lam_prev, lam, eps_lambda)
            if psi(lam_plus) > 0.0:
                lam_plus = lam_last_feas_right
            x_plus = y + lam_plus * v
            break
        lam_prev = lam
        lam = min(lam * 2.0 if lam > 0.0 else init_step, lam_right_h)
        k += 1
        if k > max_expand:
            lam_plus = lam
            x_plus = y + lam_plus * v
            break

    # Esquerda
    lam_prev = 0.0
    lam_last_feas_left = 0.0
    lam = -min(init_step, abs(lam_left_h) if lam_left_h != -math.inf else init_step)
    k = 0
    while True:
        val = psi(lam)
        if val <= 0.0:
            lam_last_feas_left = lam
        if lam <= lam_left_h + 0.0:
            if val <= 0.0:
                lam_minus = lam
                x_minus = y + lam_minus * v
                break
            else:
                lam_minus = bisect_zero(psi, lam_prev, lam, eps_lambda)
                if psi(lam_minus) > 0.0:
                    lam_minus = lam_last_feas_left
                x_minus = y + lam_minus * v
                break
        if val > 0.0:
            lam_minus = bisect_zero(psi, lam_prev, lam, eps_lambda)
            x_minus = y + lam_minus * v
            break
        lam_prev = lam
        lam = -min(
            abs(lam) * 2.0 if lam < 0.0 else init_step,
            abs(lam_left_h) if lam_left_h != -math.inf else abs(lam) * 2.0,
        )
        k += 1
        if k > max_expand:
            lam_minus = lam
            x_minus = y + lam_minus * v
            break

    return float(lam_minus), float(lam_plus), x_minus, x_plus

# ============================================================
# Utilidades: inversa da raiz quadrada via autovalores
# ============================================================

def invsqrt_and_sqrt(mat: np.ndarray, delta: float) -> Tuple[np.ndarray, np.ndarray]:
    """Dado Sigma, retorna Y=(Sigma+delta I)^{-1/2} e Yinverse=(Sigma+delta I)^{1/2}."""
    A = 0.5 * (mat + mat.T)
    A = A + delta * np.eye(A.shape[0])
    w, U = np.linalg.eigh(A)
    w = np.clip(w, 1e-18, None)
    inv_sqrt = U @ np.diag(1.0 / np.sqrt(w)) @ U.T
    sqrt = U @ np.diag(np.sqrt(w)) @ U.T
    return inv_sqrt, sqrt

# ============================================================
# Algoritmo principal: RCP-H&R-PLSI clássico
# ============================================================

@dataclass
class Params:
    N: int = 20  # amostras por iteração
    eps_lambda: float = 1e-9  # tolerância na bisseção de lambda
    eps_opt: float = 1e-10  # tolerância de melhora do objetivo
    delta_iso: float = 1e-3  # regularização para isotropização
    max_iters: int = 200  # iterações externas
    # --- parâmetros do supremo externo (obrigatório) ---
    eps_q: float = 1e-6  # colchão de segurança da malha/medição
    m_grid: int = 1024  # malha base quando não adaptativa
    adaptive_grid: bool = True  # manter h(|α|+e) <= eps_q
    quantum_backend: object | None = None  # backend com max_on_grid(f,m)->(i_max,f_max)

def rcp_hr_plsi_classico(
    x0: np.ndarray, *, rng: Optional[random.Random] = None, params: Optional[Params] = None
) -> Tuple[np.ndarray, float]:
    """Algoritmo RCP-H&R-PLSI clássico (sem 0<=gamma<=Gamma), usando supremo EXTERNO.

    φ(x) = S_hat(α,β;ε_q) − γ, onde S_hat vem de dh_oracle.sup_abs_d_dh.
    """
    if params is None:
        params = Params()
    if rng is None:
        rng = random

    alpha0, beta0 = float(x0[0]), float(x0[1])

    # Usa backend CLÁSSICO para construir y0 de forma determinística
    bk_save = params.quantum_backend
    params.quantum_backend = _ClassicalGridBackend()
    base_classic = sup_external(alpha0, beta0, params)
    params.quantum_backend = bk_save

    # Colchão ABSOLUTO generoso + proporcional (garante interioridade mesmo com ruído)
    safety_abs = 0.10          # 0.10 de folga absoluta em γ (ajustável)
    safety_rel = 20.0 * max(params.eps_q, 1e-3)
    delta0 = max(1e-3 * max(1.0, base_classic), safety_abs, safety_rel)

    gamma0 = base_classic + delta0
    y0 = np.array([alpha0, beta0, gamma0], dtype=float)

    phi = make_phi_quantum(params)

    # Se, ainda assim, φ(y0) não for negativa, infla γ até entrar
    p0 = phi(y0)
    attempts = 0
    while p0 > -1e-9 and attempts < 6:
        gamma0 += max(delta0, safety_abs)
        y0[2] = gamma0
        p0 = phi(y0)
        attempts += 1

    assert phi(y0) < 0.0 + 1e-12, "Fase I falhou: y0 não é interior (phi<=0)."

    tau = float(gamma0)

    c = np.array([0.0, 0.0, 1.0])
    melhor = float("inf")
    z_best = y0.copy()

    Y = np.eye(3)
    Y_inv = np.eye(3)

    y_curr = y0.copy()

    for k in range(params.max_iters):
        viol = phi(y_curr)
        if viol > 1e-12:
            y_curr = y_curr.copy()
            y_curr[2] = sup_external(float(y_curr[0]), float(y_curr[1]), params) + max(
                1e-12, 1e-9 * (1.0 + abs(viol))
            )
        if y_curr[2] > tau:
            tau = float(y_curr[2])
        xs_minus: List[np.ndarray] = []
        xs_plus: List[np.ndarray] = []
        ys: List[np.ndarray] = []

        yj = y_curr.copy()
        for j in range(params.N):
            w = np.array([rng.gauss(0.0, 1.0) for _ in range(3)], dtype=float)
            u = Y_inv @ w
            nrm = float(np.linalg.norm(u))
            if nrm == 0.0:
                continue
            v = u / nrm

            lam_minus, lam_plus, x_minus, x_plus = boundary_oracle(
            yj, v, tau, phi, eps_lambda=params.eps_lambda
            )


            lam = rng.uniform(lam_minus, lam_plus)
            yj = yj + lam * v

            xs_minus.append(x_minus)
            xs_plus.append(x_plus)
            ys.append(yj.copy())

        candidates = ys + xs_minus + xs_plus
        feas = [z for z in candidates if (phi(z) <= 1e-12 and z[2] <= tau + 1e-12)]
        if len(feas) == 0:
            feas = [y_curr.copy()]
        costs = [float(c @ z) for z in feas]
        order = np.argsort(costs)
        zk = feas[int(order[0])].copy()
        zpk = (
            feas[int(order[1])].copy() if len(order) > 1 else feas[int(order[0])].copy()
        )

        val = float(c @ zk)
        if val < melhor - params.eps_opt:
            melhor = val
            z_best = zk.copy()

        xs_all = xs_minus + xs_plus
        xs_mat = np.stack(xs_all, axis=0)
        mk = np.mean(xs_mat, axis=0)
        diffs = xs_mat - mk
        Sigma = (diffs.T @ diffs) / xs_mat.shape[0]
        Y, Y_inv = invsqrt_and_sqrt(Sigma, params.delta_iso)

        tau = min(tau, float(zpk[2]))
        y_curr = zk.copy()
        if y_curr[2] > tau:
            tau = float(y_curr[2])

    return z_best, melhor

# ============================================================
# Execução direta e utilitários de validação
# ============================================================

def main() -> None:
    rng = random.Random(42)

    x0 = np.array([1.0, 0.0, 0.0], dtype=float)

    # --------- MODO SEGURO PARA TESTE (evita m imenso e OOM) ---------
    params = Params(N=20, max_iters=30, eps_opt=1e-8)
    params.quantum_backend = QiskitGroverSetBackend(shots=256, c_factor=0.7, seed=123)

    # malha fixa moderada e colchão mais folgado
    params.adaptive_grid = False
    params.m_grid = 256
    params.eps_q = 1e-3
    # ------------------------------------------------------------------

    z_star, val = rcp_hr_plsi_classico(x0, rng=rng, params=params)

    alpha, beta, gamma = float(z_star[0]), float(z_star[1]), float(z_star[2])
    print("Solução aproximada (alpha, beta, gamma):", (alpha, beta, gamma))
    print("Valor do objetivo (gamma):", val)

    cand = [(0.0, alpha * 0.0 + beta - math.e**0.0), (1.0, alpha + beta - math.e)]
    if alpha > 0.0 and 0.0 <= math.log(alpha) <= 1.0:
        t_star = math.log(alpha)
        cand.append((t_star, alpha * t_star + beta - alpha))
    errs = [(t, abs(r)) for (t, r) in cand]
    print("Erros candidatos (|alpha t + beta - e^t|):", errs)
    S_hat = sup_external(alpha, beta, params)
    print("S_hat(alpha,beta) ≈", S_hat)

    if alpha > 0.0 and 0.0 <= math.log(alpha) <= 1.0:
        t_star = math.log(alpha)
        err0 = beta - 1.0
        err1 = alpha + beta - math.e
        err_star = alpha * t_star + beta - alpha
        print("Equioscilação (err0, err*, err1):", (err0, err_star, err1))


if __name__ == "__main__":
    main()

