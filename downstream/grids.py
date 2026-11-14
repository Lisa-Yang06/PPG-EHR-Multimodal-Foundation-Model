"""The search grids and fold counts every torch model in this package shares."""

# The projection width used by every embedding-level model here.
N_COMPONENTS = 128

# Weight decay on the head is what plays the role sklearn's C played. The
# ceiling is 30 rather than 100 because of the rho guard below: at the head's
# learning rate, 100 is unreachable, and a grid whose top entries only some
# models can reach makes those models incomparable.
WD_GRID = [1e-2, 1e-1, 1.0, 10.0, 30.0]
WD_GATE_GRID = WD_GRID
WD_HEAD_GRID = WD_GRID

# Searched jointly with the decay, never alone: rho = lr*wd is what is capped,
# so the reachable decays differ by rate.
LR_GRID_E = [3e-3, 1e-2, 3e-2]

# AdamW's decoupled decay multiplies the old weight by (1 - lr*wd) every step,
# so lr*wd is the per-step shrinkage. Above 1 the multiplier goes negative (the
# weight flips sign each step) and above 2 it also grows, which is the
# divergence boundary rather than a sensible regularisation range. 0.3 means at
# most a 30% cut per step.
MAX_RHO = 0.3

# Inner folds for hyperparameter selection, inside each outer fold.
N_INNER_E = 3
