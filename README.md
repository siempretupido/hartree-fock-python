# Hartree-Fock Python

This repository contains a small Hartree-Fock (RHF) implementation in Python, including analytic s-type Gaussian integrals, SCF, and nuclear gradients. It includes input examples and a workflow that supports both on-the-fly integral evaluation and reading precomputed integrals/derivatives from extended input files.

## Requirements

- Python 3.9+
- `numpy`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main program with a standard or extended input file:

```bash
python HF_inputs_outputs_debugging/src/main.py HF_inputs_outputs_debugging/inputs/h2.input
python HF_inputs_outputs_debugging/src/main.py HF_inputs_outputs_debugging/inputs/h2_631G_extended.input
```

Standard inputs compute integrals on the fly. Extended inputs read precomputed integrals and derivatives for SCF + gradients.

## Project layout

- `HF_inputs_outputs_debugging/src`: core source code (integrals, SCF, gradients, input parsing, and `main.py`).
- `HF_inputs_outputs_debugging/inputs`: sample input files (standard and extended).
- `HF_inputs_outputs_debugging/outputs`: sample output files.
- `HF_inputs_outputs_debugging/debugging`: debug output snapshots.

There is also a Jupyter notebook with detailed explanations and formulas:
- `HF_inputs_outputs_debugging/src/HF_online_project.ipynb`
