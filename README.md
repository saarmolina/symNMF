# Symmetric Non-negative Matrix Factorization (symNMF) Clustering

This project implements a clustering algorithm based on symmetric Non-negative Matrix Factorization (symNMF) and compares its performance with K-means clustering. The implementation includes both Python and C components, with Python C API integration.

## Overview

The symNMF algorithm performs clustering by:
1. Creating a similarity matrix from input data points
2. Computing a diagonal degree matrix
3. Calculating normalized similarity
4. Finding a non-negative matrix H that minimizes ||W - HH^T||²ᵣ
5. Deriving final clusters from matrix H

## Project Structure

```
.
├── symnmf.py         # Python interface
├── symnmf.c          # C implementation
├── symnmf.h          # C header file
├── symnmfmodule.c    # Python C API wrapper
├── analysis.py       # Algorithm analysis & comparison
├── setup.py          # Build configuration
└── Makefile          # Build script
```

## Requirements

- Python 3.x
- NumPy
- scikit-learn (for analysis.py)
- GCC compiler
- Make

## Installation

1. Clone the repository
2. Build the C extension:
```bash
python3 setup.py build_ext --inplace
```
3. Build the C executable:
```bash
make
```

## Usage

### Python Interface

```bash
python3 symnmf.py k goal input_file.txt
```

Parameters:
- `k`: Number of clusters (integer < N)
- `goal`: One of the following:
  - `symnmf`: Perform full symNMF algorithm and output H
  - `sym`: Calculate similarity matrix
  - `ddg`: Calculate diagonal degree matrix
  - `norm`: Calculate normalized similarity matrix
- `input_file.txt`: Path to input data file

Example:
```bash
python3 symnmf.py 2 symnmf input_1.txt
```

### C Interface

```bash
./symnmf goal input_file.txt
```

Parameters:
- `goal`: `sym`, `ddg`, or `norm`
- `input_file.txt`: Path to input data file

Example:
```bash
./symnmf sym input_1.txt
```

### Analysis

Compare symNMF with K-means clustering:
```bash
python3 analysis.py k input_file.txt
```

Parameters:
- `k`: Number of clusters
- `input_file.txt`: Path to input data file

## Input Format

The input file should contain N data points, with each point represented as a vector of floating-point numbers. Each vector should be on a separate line, with values separated by commas.

## Output Format

All outputs are formatted to 4 decimal places, with each row on a separate line and values separated by commas.

## Error Handling

In case of any error, the program will print "An Error Has Occurred" and terminate.

## Implementation Notes

- Random seed is set to 1234 for reproducibility
- Convergence parameters:
  - ε = 1e-4
  - max_iter = 300
- All vector elements use double precision in C and float in Python
- Memory management follows C best practices with proper allocation/deallocation
- Code is compiled with strict warning flags: -ansi -Wall -Wextra -Werror -pedantic-errors

## Limitations

- Input validation is not implemented
- All data points must be unique
- The program must run on Nova environment

## Reference

Based on:
Kuang, D., Ding, C., & Park, H. (2012). Symmetric nonnegative matrix factorization for graph clustering. In Proceedings of the 2012 SIAM International Conference on Data Mining (SDM) (pp. 106-117). Society for Industrial and Applied Mathematics.
