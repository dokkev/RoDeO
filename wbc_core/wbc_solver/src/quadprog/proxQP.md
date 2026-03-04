# ProxQP API Reference (for WBIC integration)

## Problem Formulation

```
min  0.5 * x^T H x + g^T x
s.t. A x = b          (equality)
     l <= C x <= u     (inequality, box)
```

- H: (d x d) symmetric PSD
- A: (n_eq x d), b: (n_eq)
- C: (n_in x d), l: (n_in), u: (n_in)
- No upper bound → pass `std::nullopt` for u

## Dense Backend API

### Include
```cpp
#include <proxsuite/proxqp/dense/dense.hpp>
using namespace proxsuite::proxqp;
```

### Constructor
```cpp
dense::QP<double> qp(d, n_eq, n_in);
```

### Init (first call)
```cpp
qp.init(H, g, A, b, C, l, std::nullopt);
// Optional params: compute_preconditioner, rho, mu_eq, mu_in
// qp.init(H, g, A, b, C, l, std::nullopt, true, 1.e-6, 1.e-3, 1.e-1);
```

### Update (subsequent calls, same dimensions)
```cpp
qp.update(H, g, A, b, C, l, std::nullopt);
// Optional: update_preconditioner (default: false)
```

### Solve
```cpp
qp.solve();
// With warm start:
// qp.solve(x_guess, y_guess, z_guess);
```

### Results
```cpp
Eigen::VectorXd x_sol = qp.results.x;   // primal solution
Eigen::VectorXd y_sol = qp.results.y;   // equality multipliers
Eigen::VectorXd z_sol = qp.results.z;   // inequality multipliers

// Status check
qp.results.info.status == QPSolverOutput::PROXQP_SOLVED

// Diagnostics
qp.results.info.iter       // total iterations
qp.results.info.pri_res    // primal residual
qp.results.info.dua_res    // dual residual
qp.results.info.objValue   // objective value
qp.results.info.setup_time // microseconds
qp.results.info.solve_time // microseconds
```

## Settings

```cpp
qp.settings.eps_abs = 1.e-5;        // absolute stopping criterion
qp.settings.eps_rel = 0;            // relative stopping criterion
qp.settings.verbose = false;
qp.settings.max_iter = 10000;
qp.settings.max_iter_in = 1500;
qp.settings.compute_timings = false;
qp.settings.initial_guess = InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
```

### Initial Guess Options
| Enum | Behavior |
|------|----------|
| `NO_INITIAL_GUESS` | Start from zero |
| `EQUALITY_CONSTRAINED_INITIAL_GUESS` | Solve equality-only relaxation (default) |
| `WARM_START_WITH_PREVIOUS_RESULT` | Reuse last qp.results (best for real-time) |
| `WARM_START` | User provides x,y,z via solve(x,y,z) |
| `COLD_START_WITH_PREVIOUS_RESULT` | Prior solution but reset proximal params |

## Solver Status Enum
```cpp
QPSolverOutput::PROXQP_SOLVED
QPSolverOutput::PROXQP_MAX_ITER_REACHED
QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE
QPSolverOutput::PROXQP_DUAL_INFEASIBLE
QPSolverOutput::PROXQP_NOT_RUN
```

## Convention Mapping: wbc_core → ProxQP

Current wbc_core uses `C*x >= l` (QuadProg++ convention).
ProxQP uses `l <= C*x <= u`.

**Mapping**: Pass `l` as lower bound, `std::nullopt` as upper bound → equivalent to `C*x >= l`.
No sign flip needed — conventions are compatible when u = +∞.

## Dimension Change Handling

If QP dimensions change between ticks, must create a new QP object:
```cpp
if (!qp_ || qp_->model.dim != dim || qp_->model.n_eq != n_eq || qp_->model.n_in != n_in) {
    qp_ = std::make_unique<dense::QP<double>>(dim, n_eq, n_in);
    qp_->settings.eps_abs = 1e-5;
    qp_->settings.verbose = false;
    qp_->settings.initial_guess = InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
    qp_->init(H_, g_, A_, b_, C_, l_, std::nullopt);
} else {
    qp_->update(H_, g_, A_, b_, C_, l_, std::nullopt);
}
qp_->solve();
```

## CMake Integration
```cmake
find_package(proxsuite REQUIRED)
target_link_libraries(target PUBLIC proxsuite::proxsuite-vectorized)
```

`proxsuite-vectorized` enables SIMD (AVX2/AVX512) via SIMDE.

## Reference Implementation Pattern (rpc_source/rpc2/wbic.cpp)
```cpp
// dense::QP<double> qp(dim, n_eq, n_ineq);
// qp.settings.eps_abs = eps_abs;
// qp.settings.initial_guess = InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
// qp.settings.verbose = false;
// qp.init(H_, nullopt, A_, b_, C_, l_, nullopt);
// qp.solve();
// Eigen::VectorXd qp_sol = qp.results.x;
```

## Performance Notes
- Dense backend: use when problem < 1000 variables or density > 10%
- Sparse backend: for large sparse problems (not needed for WBIC, typically < 30 vars)
- First solve includes factorization → solve_time includes init factorization
- Warm-start significantly reduces iterations for sequential QP solves (real-time control)
