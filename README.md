# Quantum Error Correction Module

Quantum error correction code implementations with spectral analysis of superoperators. Implements three-qubit bit flip, five-qubit surface, and thirteen-qubit surface codes.

## Installation

### Local Setup

```bash
git clone <repository-url>
cd quantum_error_correction
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### HPC Setup (Rangpur)

```bash
ssh s1234567@rangpur.compute.eait.uq.edu.au
git clone <repository-url>
cd quantum_error_correction

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt --no-cache-dir
```

## Usage

### Local Execution

```bash
python main.py --code_type three_qubit --tricky True
python main.py --code_type five_qubit_surface --tricky False
python main.py --code_type thirteen_qubit_surface --tricky False
```

### HPC Execution

**Single job:**
```bash
export CODE=thirteen_qubit_surface TRICKY=False
sbatch run_qec.sh
```

**Multiple jobs:**
```bash
bash submit_all.sh
```

**Monitor:**
```bash
squeue -u $USER
cat logs/qec_*.out
```

## Arguments

- `--code_type`: Error correction code (`three_qubit`, `five_qubit_surface`, `thirteen_qubit_surface`)
- `--tricky`: Use alternative superoperator construction (`True`/`False`)
- `--device`: Compute device (`auto`, `cuda`, `mps`, `cpu`)

## Structure

```
quantum_error_correction/
├── quantum_states.py           # Pauli operators and basis states
├── error_codes.py              # Error correction code classes
├── quantum_error_correction.py # Superoperator construction
├── utils.py                    # Utility functions
├── analysis.py                 # Spectral analysis
├── plotting.py                 # Visualization
├── main.py                     # Entry point with argparse
├── run_qec.sh                  # Slurm template
└── submit_all.sh               # Batch submission script
```

## Adding New Codes

1. Create class in `error_codes.py` inheriting from `ErrorCorrectionCode`
2. Implement `create_stabilizers()` and `create_recovery_map()`
3. Add to `_initialize_code()` dictionary in `quantum_error_correction.py`

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy
- Matplotlib

## HPC Notes

- Never run computations on login node
- Set reasonable time limits (≤1 day)
- Monitor jobs with `squeue -u $USER`
- Cancel jobs with `scancel <jobid>`
- Requires UQ VPN when off-campus