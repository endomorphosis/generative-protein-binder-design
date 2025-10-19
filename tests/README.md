# Tests Directory

This directory contains test files for the protein binder design project.

## Test Files

- `test_attention_solutions.py` - Tests for attention mechanisms on ARM64 Blackwell GB10
- `test_flash_attention_working.py` - Tests for Flash Attention functionality on ARM64 Blackwell GB10 GPU

## Running Tests

To run individual test files:

```bash
python3 tests/test_attention_solutions.py
python3 tests/test_flash_attention_working.py
```

Note: These tests require CUDA-capable GPUs to run successfully.
