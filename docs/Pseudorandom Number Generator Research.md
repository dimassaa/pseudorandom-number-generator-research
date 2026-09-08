# Technical Document: Pseudorandom Number Generator Research

**Project:** Analysis, visualization, and prediction of sequences from popular pseudorandom number generators (PRNGs) without using machine learning.  
**Target audience:** developers, researchers, cryptography and statistics enthusiasts reading the GitHub repository.  
**Environment:** Linux/WSL, Python 3.9+, libraries: `numpy`, `matplotlib`, `scipy`, `time`, `random`, `itertools`.

---

### 1. Introduction

Pseudorandom number generators lie at the heart of simulations, games, and cryptography. However, the quality of their randomness varies significantly: some pass strict statistical tests, while others show clear patterns. The goal of this project is to perform a comparative analysis of several widely known PRNGs, identify their weaknesses, visualize statistical defects, and demonstrate attacks that allow predicting future outputs based on observations.

Within this project we will:

- implement and/or use built-in generators (LCG, Mersenne Twister, xorshift, V8 Math.random);
- apply a battery of statistical tests (uniformity, autocorrelation, spectral analysis);
- implement state recovery algorithms for LCG and MT19937;
- emulate JS Math.random (V8 engine) and analyze its predictability;
- compare generator performance with large output volumes.

All computations are performed locally, resource-intensive operations are allowed, and the final report is oriented toward reproducibility and high-quality visualizations.

---

### 2. Overview of Generators

#### 2.1 Linear Congruential Generator (LCG)

The LCG is described by the recurrence:
```
X_{n+1} = (a * X_n + c) mod m
```
where `a` is the multiplier, `c` the increment, `m` the modulus, and `X_0` the initial seed.  
The LCG was widely used in standard libraries (C `rand()`, glibc, Java `Random` before certain versions). It has a well-known flaw: points `(X_n, X_{n+1})` lie on a small number of hyperplanes, easily visualized in 2D.

**In this project we implement several LCG variants:**

- ANSI C parameters (`a=1103515245, c=12345, m=2^31`).
- Numerical Recipes parameters (`a=1664525, c=1013904223, m=2^32`).
- glibc `rand()` parameters (`a=1103515245, c=12345, m=2^31`, but outputting higher bits which are considered better).
- A custom “bad” LCG with a small modulus (e.g., `m=101`) for clear demonstration of planes.

#### 2.2 Mersenne Twister (MT19937)

The Mersenne Twister is one of the most widespread PRNGs, used in Python (`random`), R, MATLAB, etc. It has a period of 2^19937−1 and good statistical properties, but it is not cryptographically secure: by observing 624 consecutive 32-bit outputs, one can fully recover the internal state and predict all future numbers.

We will use the built-in `random` module as a source, but for the attack we will implement our own `MT19937` class and the inverse tempering operation.

#### 2.3 JS Math.random (V8)

In modern browsers (Chrome, Node.js), `Math.random()` is implemented using the **xorshift128+** algorithm (two 64-bit states, 128 bits total). The output is a 53-bit floating-point number (top 26 bits discarded). This generator is also vulnerable to state recovery given enough outputs.

For analysis, we will emulate the xorshift128+ algorithm in Python according to the open V8 description and check whether state recovery or prediction of future values is possible.

#### 2.4 Additional Generators (for comparison)

- **xorshift32** (simple, fast, used for statistical comparison).
- **PCG32** (Permuted Congruential Generator) — modern fast generator with good statistical properties; we implement a simplified version.
- Optionally: **Middle Square** (historical generator with visible defects).

---

### 3. Analysis Methodology

#### 3.1 Sequence Generation

For each generator, sequences of length `N` (default `N = 10^6`) are generated as 32-bit integers (or normalized to [0,1)). All tests are run on multiple independent sequences (different seeds) to assess stability.

#### 3.2 Statistical Tests

1. **Chi-square test for uniformity.**  
   Divide the range into `k` equal bins (e.g., `k = 1000`). Compute the chi-square statistic and p-value. A good generator should yield p-value in [0.01, 0.99].

2. **Autocorrelation.**  
   Compute Pearson correlation coefficient between the sequence and its shifted version at lag `t` (t = 1, 2, 5, 10). For an ideal PRNG the coefficient should be close to 0 (± 1/√N).

3. **Spectral test (for LCG).**  
   Plot 2D graph of pairs `(X_n, X_{n+1})`. LCGs with bad parameters show parallel lines; good generators should fill the space uniformly.

4. **Runs test.**  
   Check independence of bits or digits: count the number of runs (consecutive identical bits) and compare with expected for a random sequence.

5. **Additional visualizations:**
   - Histogram of value distribution.
   - Autocorrelation plot by lag.
   - Comparative performance chart (see Section 5).

All tests are implemented using `numpy` and `scipy.stats` for accuracy and speed.

#### 3.3 Implementation of Tests

Each test is a function taking an array of integers and returning a dictionary with metrics (statistic, p-value, comment). Results are saved to JSON for later reporting.

---

### 4. Attacks and Prediction

#### 4.1 LCG: Parameter Recovery and Prediction

Case 1: **modulus `m` is known** (e.g., power of two).  
Observe three consecutive values `X_0, X_1, X_2`. Then compute:
```
a = (X_2 - X_1) * inverse(X_1 - X_0, m) mod m
c = (X_1 - a * X_0) mod m
```
After that, predict all subsequent values.

Case 2: **modulus unknown.**  
Use the method of finding `m` through differences:  
`T_n = X_{n+1} - X_n`. The modulus `m` divides the greatest common divisor of `T_n * T_{n+2} - T_{n+1}^2` for several `n`. In practice 4–5 outputs are enough to find `m` (or a small multiple), then proceed as in Case 1.

We implement both variants and verify prediction accuracy on 1000 subsequent numbers.

#### 4.2 Mersenne Twister: Full State Recovery

Algorithm:

1. Obtain 624 consecutive 32-bit integers from `random.getrandbits(32)` (or use our own MT19937 implementation).
2. For each number, perform inverse tempering (undo bitwise operations in reverse order).
3. The resulting 624 numbers become the state array `mt` (first 624 elements).
4. Set index `index = 624` and reproduce subsequent numbers using the normal generation algorithm.

**Inverse tempering** includes inverting:
- `y ^= y >> 11`
- `y ^= (y << 7) & 0x9d2c5680`
- `y ^= (y << 15) & 0xefc60000`
- `y ^= y >> 18`

Each operation is invertible bit by bit; we implement functions `inverse_right_shift_xor` and `inverse_left_shift_xor_mask`.

Verification: after recovery, compare the state with `random.getstate()[1][:624]` — should match (considering byte order). Then generate the next 1000 numbers and compare with actual — 100% match.

#### 4.3 JS Math.random (xorshift128+)

We emulate the algorithm:

- State: two 64-bit unsigned integers `s0, s1`.
- Function `xorshift128plus()`:
  - `x = s0`, `y = s1`
  - `s0 = y`
  - `x ^= x << 23`
  - `s1 = x ^ y ^ (x >> 17) ^ (y >> 26)`
  - return `s1 + y` (mod 2^64).
- JavaScript's `Math.random()` uses the top 53 bits of the result: `(result >> 11) / 2^53`.

We can generate sequences with a known seed (state initialization). To analyze predictability, we attempt to solve the inverse problem: given a series of 64-bit results (or 53-bit floats) — can we recover `s0, s1`?

For xorshift128+ there is a known attack requiring several outputs and solving a system of non-linear equations (using SMT solvers like Z3). In this project without ML we can either:

- Implement an attack using `z3-solver` (optional dependency), or
- Demonstrate that one output is insufficient to recover state, but with enough observations a system can be built and solved by brute force (for simplified xorshift with smaller state).

For simplicity, we consider a **simplified version**: xorshift32 (32-bit state) and xorshift64. For xorshift32 we can brute-force all 2^32 states in acceptable time (using C or numba) — this clearly shows the generator is not cryptographically secure. For xorshift64, brute force is long, but we can apply a linear algebra method (since xorshift is linear over bits) to recover state from several outputs.

Thus, for JS Math.random we demonstrate emulation and note the vulnerability, but full attack on V8 may be described as possible but out of scope of current implementation (or added as a separate module using Z3).

#### 4.4 Expected Results

- LCG: parameter recovery and prediction with 100% accuracy.
- MT19937: state recovery and prediction with 100% accuracy.
- xorshift32: full state brute-force and prediction with 100% accuracy.
- xorshift64: recovery via linear system (if feasible) or note on difficulty.
- V8 xorshift128+: emulation confirms identical output for same seed; prediction possible with SMT solver (described but not fully implemented).

---

### 5. Performance

Goal — compare generation speed for each generator under identical conditions.

**Methodology:**

- For each generator, measure time to generate `N = 10^7` numbers (32-bit integers) in a Python loop.
- Use `time.perf_counter()` for timing, repeat 3 times, take average and standard deviation.
- For built-in `random`, use `random.getrandbits(32)`.
- For custom implementations — pure Python; optionally add acceleration via `numba` for fair comparison (but note JIT compilation).
- Visualize results as a bar chart with logarithmic scale (differences may be large).

**Expectations:**

- Built-in C implementations (Python `random`) will be faster than pure Python emulations.
- LCG and xorshift in Python will be comparable in speed.
- Brute force for xorshift32 will be very slow in pure Python — use C extension or numba, or reduce the state size (e.g., truncated xorshift16 for demonstration).

---

### 6. Architecture and Implementation

Repository structure:

```
prng-analysis/
├── README.md
├── requirements.txt
├── src/
│   ├── generators/
│   │   ├── lcg.py
│   │   ├── mt19937.py
│   │   ├── xorshift.py
│   │   ├── v8_random.py
│   │   └── pcg.py
│   ├── tests/
│   │   ├── statistical_tests.py
│   │   └── visualization.py
│   ├── attacks/
│   │   ├── lcg_attack.py
│   │   ├── mt19937_attack.py
│   │   └── xorshift_attack.py
│   ├── benchmarks.py
│   └── main.py
├── results/
│   ├── figures/
│   └── metrics.json
└── docs/
    └── report.md
```

**Module descriptions:**

- `generators/lcg.py` — LCG classes with various parameters, methods `next_int()`, `next_float()`.
- `generators/mt19937.py` — own MT19937 implementation, identical to standard, including seed initialization.
- `generators/xorshift.py` — xorshift32, xorshift64, xorshift128+ (for V8 emulation).
- `generators/v8_random.py` — emulation of V8 Math.random() (xorshift128+ with double conversion).
- `generators/pcg.py` — simplified PCG32.
- `tests/statistical_tests.py` — chi-square, autocorrelation, runs test, spectral test functions.
- `tests/visualization.py` — generation of all plots, saving PNG.
- `attacks/lcg_attack.py` — parameter recovery and prediction.
- `attacks/mt19937_attack.py` — inverse tempering, state recovery.
- `attacks/xorshift_attack.py` — brute force for xorshift32, linear algebra for xorshift64 (optional).
- `benchmarks.py` — performance measurements.
- `main.py` — orchestration: run tests, attacks, benchmarks, save results.

**Dependencies:** `numpy`, `matplotlib`, `scipy`, `z3-solver` (optional, for advanced attack on xorshift128+). For brute-force acceleration, add `numba` or write critical parts in C via `ctypes`.

---

### 7. Execution Plan

1. **Week 1:** Implement generators (LCG, MT19937, xorshift, V8, PCG). Write unit tests to verify correctness (compare with reference output sequences).
2. **Week 2:** Implement statistical tests and visualizations. Run on all generators, collect metrics.
3. **Week 3:** Implement attacks on LCG and MT19937, verify with random seeds.
4. **Week 4:** Emulate V8 and analyze predictability, implement brute force for xorshift32, assess feasibility for xorshift64/128.
5. **Week 5:** Performance benchmarking, optimize critical sections if needed.
6. **Week 6:** Compile all results into report (`docs/report.md`), polish figures, write README with reproduction instructions.

---

### 8. Expected Results and Success Criteria

- **LCG:** 2D plot clearly shows structure (planes). Parameter recovery successful for known and unknown modulus (with caveats for large modulus). Prediction of next numbers accurate.
- **MT19937:** Inverse tempering correct, state recovered, next 1000 numbers match actual.
- **V8 xorshift128+:** Emulation generates same sequence as Node.js (with same seed). Demonstration that generator is not cryptographically secure (descriptively or with Z3 if implemented).
- **Statistics:** Histograms look uniform for all generators except intentionally bad LCG. Autocorrelation near zero. Chi-square p-value within acceptable limits.
- **Performance:** Comparative timings obtained, conclusions drawn about suitability in high-load systems.

---

### 9. Risks and Limitations

- **Difficulty of xorshift128+ attack without SMT solver:** full state recovery in reasonable time may be impossible. In that case, limit to describing the attack and demonstrating on simplified versions.
- **Brute force xorshift32 in Python:** 2^32 states ≈ 4 billion, in pure Python would take hours. Use `numba` (JIT) or C function via `ctypes` to fit in minutes.
- **Differences in JS Math.random implementations:** different V8 versions may have used different algorithms (xorshift128+ since Chrome 49). We fix this version.
- **Computation volume for statistical tests:** 10^6 numbers sufficient for most tests, but spectral analysis may require fewer (e.g., 10^5 for clarity). Adjustable via parameters.

---

### 10. Conclusion

This project combines theoretical analysis, practical programming, and visualization, making it valuable both for education and for demonstrating vulnerabilities of standard PRNGs. Results will benefit anyone interested in randomness, simulations, and cryptography.
