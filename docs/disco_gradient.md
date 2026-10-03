# Exact gradient of the weighted distance correlation

This note derives the gradient of the weighted squared distance correlation
`dCorr2_w(s, m)` with respect to the scores `s`, as implemented by
`vikshep_stats::dcorr::dcorr2_grad` and specified in `spec/VDS-1.md`
section 16.2. The DisCo training objective (section 17.4) uses it for the
dependence term

```
L = wBCE(yhat, y) + lambda * dCorr2_w(yhat, m | background)
```

with `m` (the protected variable) fixed and `s = yhat` on background events.

## Definitions

Events `i = 1..n` carry weights `w_i >= 0`; `v_i = w_i / sum_l w_l`, so
`sum_i v_i = 1`. For a variable `x`:

```
a_ij = |x_i - x_j|
r_i  = sum_j v_j a_ij              (weighted row mean; a is symmetric, so
mu   = sum_i v_i r_i                column means equal row means)
A_ij = a_ij - r_i - r_j + mu        (weighted double centering)
```

and `B` likewise for `y`. With the weighted Frobenius inner product
`<X, Y> = sum_ij v_i v_j X_ij Y_ij`:

```
dCov2(x, y)  = <A, B>
dCorr2(x, y) = dCov2(x, y) / sqrt(dCov2(x, x) dCov2(y, y))
```

## Double centering is an orthogonal projection

Write `H = I - 1 v^T`, so `(H u)_i = u_i - sum_l v_l u_l`. Then
`A = H a H^T`, because

```
(H a H^T)_ij = a_ij - sum_l v_l a_lj - sum_l a_il v_l + sum_lk v_l v_k a_lk.
```

`H` is idempotent,
`H H = I - 2 (1 v^T) + 1 (v^T 1) v^T = I - 1 v^T = H` (using `v^T 1 = 1`),
and self-adjoint for the weighted inner product `<u, z>_v = sum_i v_i u_i z_i`:

```
<H u, z>_v = sum_i v_i u_i z_i - (v^T u)(v^T z),
```

which is symmetric in `u` and `z`. Hence the centering map
`C(X) = H X H^T` is an orthogonal projection for `<., .>`:
`<C X, Y> = <C X, C Y> = <X, C Y>`, and in particular

```
<A, B> = <C a, C b> = <a, C b> = <a, B>.                      (1)
```

So `dCov2(x, y) = sum_ij v_i v_j a_ij B_ij`: only one argument needs to be
centered.

## Derivatives of the two moments

Let `s` be the scores and `m` the protected variable, with centered
matrices `A` (from `s`) and `B` (from `m`). Define

```
D  = dCov2(s, m) = <A, B>,   Vs = dCov2(s, s) = <A, A>,   Vm = dCov2(m, m) = <B, B>.
```

`Vm` does not depend on `s`. Away from ties (`s_i != s_j` for all `i != j`)
`a_ij = |s_i - s_j|` is differentiable with

```
d a_ij / d s_k = sign(s_i - s_j) (delta_ik - delta_jk).
```

**D.** By (1), `D = <a, B>` and `B` does not depend on `s`:

```
dD/ds_k = sum_ij v_i v_j B_ij sign(s_i - s_j) (delta_ik - delta_jk)
        = v_k sum_j v_j sign(s_k - s_j) B_kj - v_k sum_i v_i sign(s_i - s_k) B_ik
        = 2 v_k sum_j v_j sign(s_k - s_j) B_kj,
```

using `B_ik = B_ki` and `sign(s_i - s_k) = -sign(s_k - s_i)`.

**Vs.** `Vs = <C a, C a>`, and `C` is linear, so with `dA = C(da)`:

```
dVs = 2 <A, C(da)> = 2 <C A, da> = 2 <A, da>      (C A = A, C self-adjoint)
dVs/ds_k = 4 v_k sum_j v_j sign(s_k - s_j) A_kj.
```

## Gradient of dCorr2

With `R = D / sqrt(Vs Vm)`:

```
dR/ds_k = (dD/ds_k) / sqrt(Vs Vm) - (D / 2) Vm (dVs/ds_k) / (Vs Vm)^(3/2)
        = (1 / sqrt(Vs Vm)) (dD/ds_k - (D / (2 Vs)) dVs/ds_k)
        = (2 v_k / sqrt(Vs Vm)) sum_j v_j sign(s_k - s_j) (B_kj - (D / Vs) A_kj).
```

This is the formula implemented, evaluated as

```
c   = D / Vs
g_k = ((2 v_k) / sqrt(Vs Vm)) * psum_j( v_j * sign(s_k - s_j) * (B_kj - c * A_kj) )
```

with `psum` the pairwise sum of section 15.2 and `A_kj`, `B_kj` recomputed
from the row means, as in the statistic itself. Cost: O(n^2) time, O(n)
memory per row.

**Conventions.** At ties `a_ij` is not differentiable; the implementation
uses `sign(0) = 0`, a valid subgradient choice. Where `dCorr2` is defined as
0 because `sqrt(Vs Vm) < 1e-12` (for example, constant scores), the
gradient is 0. A useful check: shifting every score by the same constant
leaves `R` unchanged, so `sum_k g_k = 0`, which the pairwise
antisymmetry of the formula guarantees exactly in real arithmetic.

**Chain rule in training.** For a head producing logits `z_k` with
`yhat_k = sigmoid(z_k)`, the DisCo term contributes
`lambda * g_k * yhat_k (1 - yhat_k)` to `dL/dz_k` for background events
(section 17.4).

## Verification

`crates/vikshep-stats/tests/gradient.rs` compares every component of the
analytic gradient with the central finite difference
`(R(s + h e_k) - R(s - h e_k)) / (2 h)`, `h = 1e-6`, on weighted and
unweighted samples (`n` = 40, 60, 120). The largest discrepancy is below
`1e-9` of the largest gradient component (stated tolerance `1e-6`).

## Pearson proxy (fast mode)

The public Vikshep CLI trains with a Pearson-correlation proxy
(`_pearson_dcorr2_grad` in `backend/ingest/src/vikshep_ingest/cli/_train.py`).
It is the gradient of the squared weighted Pearson correlation `r^2` of `s`
and `m`, not of `dCorr2`; it captures only linear dependence. It is kept as
`GradientMode::PearsonProxy`, labelled as a fast mode, and the same test file
checks that it matches the finite differences of `r^2` and differs from the
exact dCorr2 gradient.
