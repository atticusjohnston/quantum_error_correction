# Quantum Error Correction Module

## Structure

```
quantum_error_correction/
├── __init__.py
├── quantum_states.py       # Basic quantum states and Pauli operators
├── error_codes.py          # Error correction code implementations
├── quantum_error_correction.py  # Main QEC class
├── utils.py               # Utility functions
├── analysis.py            # Spectral analysis
├── plotting.py            # Visualization
└── main.py               # Entry point
```

## Usage

```python
from quantum_error_correction import QuantumErrorCorrection

# Three-qubit bit flip code
qec = QuantumErrorCorrection(code_type='three_qubit')

# Five-qubit code
qec = QuantumErrorCorrection(code_type='five_qubit')

# Surface code
qec = QuantumErrorCorrection(code_type='surface', distance=3)

# Build superoperator
superoperator = qec.build_superoperator(tricky=True)
```

## Adding New Codes

1. Create a new class in `error_codes.py` inheriting from `ErrorCorrectionCode`
2. Implement `create_stabilizers()` and `create_recovery_map()`
3. Add to the codes dictionary in `QuantumErrorCorrection._initialize_code()`

## Running

```bash
python main.py
```

Change `code_type` in `main.py` to test different codes.