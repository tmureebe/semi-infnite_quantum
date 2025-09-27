Aproximação uniforme de e^t por funções afins: versão clássica e versão híbrida (Dürr–Høyer via Qiskit)

Descrição
Este repositório contém duas implementações do problema de melhor aproximação uniforme de e^t em [0,1] por funções afins α t + β.
1) rcp_hr_plsi_classico.py: algoritmo clássico (RCP-H&R-PLSI) calculando o supremo em malha na CPU.
2) principal_quantico.py: versão híbrida, na qual o cálculo do máximo em malha é delegado a um backend de Dürr–Høyer (Grover) implementado com Qiskit Aer.

Arquivos principais
- rcp_hr_plsi_classico.py
- principal_quantico.py
- dh_oracle.py
- qiskit_backend_dh.py

Requisitos
- Python 3.11+
- numpy
- qiskit
- qiskit-aer

Instalação com conda (recomendado)
Opção A (environment.yml)
    conda env create -f environment.yml
    conda activate qiskit-env
Opção B (manual)
    conda create -n qiskit-env python=3.11 -y
    conda activate qiskit-env
    conda install -c conda-forge qiskit qiskit-aer numpy

Instalação com pip (alternativa)
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Execução
Versão clássica:
    python rcp_hr_plsi_classico.py
Versão híbrida:
    python principal_quantico.py

Notas
- A versão híbrida usa Qiskit Aer (simulador) e pode ser custosa em CPU; ajuste shots, c_factor e max_rounds em qiskit_backend_dh.py e N, max_iters, eps_q em principal_quantico.py.
- Para acompanhar progresso, adicione prints no loop principal.
- Use seeds para reprodutibilidade.

Licença
MIT (veja LICENSE).

Como citar

Se você utilizar este repositório em trabalhos acadêmicos, cite-o da seguinte forma.

Referência sugerida (português):
Carrijo, T. M. Aproximação uniforme de e^t por funções afins: versão clássica e versão híbrida (Dürr–Høyer via Qiskit). Repositório GitHub, versão v0.1.0, 2025. Disponível em: https://github.com/SEU_USUARIO/SEU_REPO. Acesso em: dd mmm aaaa.

BibTeX:
@software{carrijo_plsi_dh_2025,
  author  = {T. M. Carrijo},
  title   = {Aproximação uniforme de e^t por funções afins: versão clássica e versão híbrida (Dürr–Høyer via Qiskit)},
  year    = {2025},
  url     = {https://github.com//tmureebe/semi-infnite_quantum},
  version = {v0.1.0},
  note    = {Python; Qiskit; arquivos principais: rcp_hr_plsi_classico.py, principal_quantico.py, dh_oracle.py, qiskit_backend_dh.py}
}
EOF
