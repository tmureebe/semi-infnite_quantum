# qiskit_backend_dh.py
from __future__ import annotations
import math, random
from typing import Callable, Iterable, List, Optional, Tuple

from qiskit import QuantumCircuit
from qiskit_aer import Aer

class QiskitGroverSetBackend:
    """
    Backend 'max_on_grid' via Dürr–Høyer com oráculo de set-membership,
    implementando manualmente r iterações de Grover no qasm_simulator (Aer).
    NÃO há fallback clássico — o resultado pode ser subótimo.
    """

    def __init__(
        self,
        shots: int = 512,
        c_factor: float = 0.7,
        max_rounds: int = 5000,
        seed: Optional[int] = None,
        tries_per_level: int = 3,      # nº de amostragens de r por limiar y
        restarts: int = 1,             # nº de reinícios com k0 diferente
        fail_tolerance: int = 1,       # quantas falhas consecutivas permitem antes de parar
    ):
        self.backend = Aer.get_backend("qasm_simulator")
        try:
            self.backend.set_options(shots=shots)
            if seed is not None:
                self.backend.set_options(seed_simulator=seed)
        except Exception:
            pass
        self.shots = shots
        self.c_factor = c_factor
        self.max_rounds = max_rounds
        self.tries_per_level = max(1, int(tries_per_level))
        self.restarts = max(1, int(restarts))
        self.fail_tolerance = max(1, int(fail_tolerance))
        self.rnd = random.Random(seed)

    # -------- API requerida por dh_oracle.py --------
    def max_on_grid(self, f: Callable[[int], float], m: int) -> Tuple[int, float]:
        N = m + 1
        n = max(1, math.ceil(math.log2(N)))  # qubits do índice

        best_global_i, best_global_val = 0, float("-inf")

        for _ in range(self.restarts):
            # amostra inicial
            k0 = self.rnd.randrange(0, N)
            y = float(f(k0))   # limiar corrente
            best_i, best_val = k0, y

            rounds = 0
            fails = 0
            while rounds < self.max_rounds and fails < self.fail_tolerance:
                rounds += 1

                # conjunto de melhorantes com base no limiar atual (cálculo clássico do conjunto)
                improving = [k for k in range(N) if k <= m and f(k) > y]
                if not improving:
                    break

                r_max = max(0, int(self.c_factor * math.sqrt(N)))

                improved = False
                # várias amostragens de r neste mesmo limiar
                for _try in range(self.tries_per_level):
                    r = self.rnd.randrange(0, r_max + 1)
                    k_candidate = self._grover_sample_candidate(improving, n, r)
                    if k_candidate is None or k_candidate > m:
                        continue
                    val = float(f(k_candidate))
                    if val > y:
                        y = val
                        best_i, best_val = k_candidate, val
                        improved = True
                        break  # avança de nível assim que melhora

                if improved:
                    fails = 0
                else:
                    fails += 1  # não melhorou neste nível

            # guarda o melhor desta corrida
            if best_val > best_global_val:
                best_global_i, best_global_val = best_i, best_val

        return best_global_i, best_global_val

    # -------- construção do oráculo e difusor --------
    @staticmethod
    def _oracle_for_set(marked: Iterable[int], n: int) -> QuantumCircuit:
        """
        Oráculo de fase: aplica -1 em |x> se x ∈ marked.
        Para cada x marcado: X nos bits 0, Z multi-controlado, desfaz X.
        """
        qc = QuantumCircuit(n)
        marked = list(marked)
        if not marked:
            return qc
        for x in marked:
            bits = [(x >> i) & 1 for i in range(n)]  # little-endian
            for q in range(n):
                if bits[q] == 0:
                    qc.x(q)
            if n == 1:
                qc.z(0)
            else:
                qc.h(n - 1)
                qc.mcx(list(range(n - 1)), n - 1)
                qc.h(n - 1)
            for q in range(n):
                if bits[q] == 0:
                    qc.x(q)
        return qc

    @staticmethod
    def _diffuser(n: int) -> QuantumCircuit:
        qc = QuantumCircuit(n)
        qc.h(range(n))
        qc.x(range(n))
        if n == 1:
            qc.z(0)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        qc.x(range(n))
        qc.h(range(n))
        return qc

    def _grover_sample_candidate(self, marked: List[int], n: int, r: int) -> Optional[int]:
        # estado inicial uniforme
        qc = QuantumCircuit(n, n)
        qc.h(range(n))

        oracle_circ = self._oracle_for_set(marked, n)
        diffuser_circ = self._diffuser(n)

        for _ in range(r):
            qc.compose(oracle_circ, qubits=range(n), inplace=True)
            qc.compose(diffuser_circ, qubits=range(n), inplace=True)

        qc.measure(range(n), range(n))

        job = self.backend.run(qc, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        if not counts:
            return None
        bitstring = max(counts.items(), key=lambda kv: kv[1])[0]
        return int(bitstring, 2)

