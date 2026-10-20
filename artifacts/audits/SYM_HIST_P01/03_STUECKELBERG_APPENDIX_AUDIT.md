# Stueckelberg AppendixG: finite mass versus exact constraint

## H07 — exact sources and present/missing steps [PARTIAL_DERIVATION]

|Object|v2 source|v3 corresponding source|Symmetry corresponding source|
|---|---|---|---|
|tbar=t−d epsilon|file p.45 A60|file p.40 Eq97|p.44 A65|
|S_Stk|file p.45 A61|file p.40 Eq98|p.44 A66|
|Printed trace equation|file p.45 A62|file p.41 Eq99|p.44 A67|
|Asserted exact t=d epsilon|file p.46 A63|file p.41 Eq100|p.44 A68|
|Hard penalty/current alternatives|file p.46 A64|file p.41 Eq101|p.44 A69|
|Low-energy action/quarter IT|file p.46 A65|file p.41 Eq102|p.45 A70|
|Weighted epsilon NY addition|file p.46 A66|file p.41 Eq103|p.45 A71|

v1 ends at AppendixF and has no corresponding AppendixG. Equation numbering is not interchangeable: v3 uses plain97–103, and Symmetry A61–A65 belongs to other material before its AppendixG. This report follows the numbers actually printed in each file.

All three later versions provide an explicit action

```
S_Stk = integral sqrt(-g)[M_P² R(e,omega)/2
          −(m_T²/2)(t_mu−v_mu)(t^mu−v^mu)
          −(f_epsilon²/2)v_mu v^mu],  v=d epsilon.
```

They state m_T>0, f_epsilon≥0, A1–A5, discard admissible divergences, and say alpha1 in the following equation comes from the curvature trace-quadratic piece. The existence of this written action and trace equation is recovered source material. What is absent is an evaluated curvature coefficient with unambiguous match to the main action's independent alpha1, and a correct solution step.

## H08 — the written finite-mass solution is false [CONTRADICTED_AS_WRITTEN]

Grant the paper its own printed trace equation, without assuming that its coefficient/sign has already been derived from S_Stk:

```
m_T²(t_mu−v_mu)+(4alpha1/3)t_mu=0.
```

Writing c=4alpha1/3 only as an algebra abbreviation, exact solution for finite m_T is

```
t_mu = [m_T²/(m_T²+c)]v_mu,         m_T²+c != 0,
tbar_mu = −[c/(m_T²+c)]v_mu.
```

Substitution of the asserted t=v into the written equation instead leaves residual `c v_mu`. Therefore the source's exact solution holds only on exceptional branches such as c=0 or v=0, or as the stated hard-penalty limit under appropriate coefficient behavior. No such branch restriction accompanies A63/Eq100/A68. Positive m_T² by itself also does not imply uniqueness: if c=−m_T², the coefficient vanishes. For nonzero v the trace equation is then inconsistent; for v=0 it loses trace uniqueness. This is an algebraic diagnostic, not a choice of new physical coefficients.

With finite fixed c and v,

```
limit_(m_T² -> infinity) t_mu = v_mu,
tbar_mu = O(c/m_T²).
```

This explains which **limiting** enforcement the old action can support; it does not turn the finite-mass solution into the asserted exact equality. E≪m_T alone is not the same limit as m_T²≫|c|, nor does it set the residual exactly to zero. With the same epsilon as the main lock, its finite proportionality would have to match3eta=m_T²/(m_T²+c); no version derives this relation as the original main-action eta.

Every algebraic step above is checked with existing SymPy in `evidence/verify_written_algebra.py` and `algebra_checks.json`. These are scoped checks of printed equations, not a recovered physical theorem.

## H09 — actual added action and limiting domain [ASSUMPTION_CHANGED]

Expanding the given mass term yields

```
−(m_T²/2)t² + m_T² t.v −[(m_T²+f_epsilon²)/2]v².
```

It therefore specifies an additional trace-sector completion and related coefficients; it is not a proof for arbitrary independent alpha1,2,3 in the main S0 skeleton. Main S0 uses the full tensor I1, not merely the trace t², so equality of complete axial/tensor sectors also needs a map. None is supplied. The audit neither adjusts these coefficients nor inserts this completion into journal Eq13.

The alternative `L_Lambda=sqrt(-g)Lambda^mu(t_mu−v_mu)` gives tbar=0 exactly by deltaLambda. This is a legitimate **constraint-enforced step** under a different variational problem. It agrees with the hard limit's stationary constraint, not with generic finite-mass stationarity. It corresponds to3eta=1 only if epsilon/trace conventions are identical; no arbitrary-eta equivalence follows.

At finite mass, integrating an algebraic quadratic variable is not an exact functional delta imposing tbar=0. An exact delta requires an actual multiplier integration or a carefully defined constraint limit; in Lorentzian signature, a “penalty” term is oscillatory rather than automatically a positive Gaussian. The source supplies no path-integral measure/normalization/contour calculation establishing its stronger delta-function statement. The classical limiting stationary equation is the narrower conclusion used here.

A61/Eq98 has no displayed torsion derivative kinetic term beyond the curvature divergence/algebraic split, so the quoted O(partial^4/m_T²) effective action estimate is not an explicit calculation of such operators. The algebraic elimination can leave two-derivative v² terms; taking f_epsilon²→0 is not by itself proof of a nondynamical scalar or zero induced kinetic term. Holding epsilon fixed is a separate declared choice, which is not changed by this audit.

## H10 — symmetry and boundary completion does not close old gaps [CONTRADICTED_AS_WRITTEN; NOT_FOUND_IN_AVAILABLE_SOURCES]

For v2 onwards, calling tbar invariant under arbitrary projective one-form xi still requires `delta(d epsilon)=3xi`, while for a scalar it is `d(delta epsilon)`. Nonclosed xi cannot be realized. The affine shift also exits the compatible antisymmetric spin-connection space at fixed metric. These problems in A01 do not disappear because a Stueckelberg action is named. The spectator v² term is not automatically invariant under an arbitrary local shift of v either; its claimed shift symmetry epsilon→epsilon+constant is weaker than that projective transformation. No full invariance calculation is supplied.

The optional addition `(alpha/4)epsilon NY` is **not** a constant multiple of an exact form. With Q=e_A wedge T^A,

```
epsilon*dQ = d(epsilon Q) − d epsilon wedge Q.
```

The second term needs evaluation and does not follow to be a boundary merely because NY=dQ. Under strictly pure trace Q=0 it vanishes, but then it cannot create the missing nonzero C2 bulk coefficient. On unrestricted torsion it changes the variation and cannot be used as an innocuous all-purpose improvement without calculation. v2 A66, v3 Eq103 and Symmetry A71 provide no route-specific nonzero CS+ normalization derivation.

## H11 — normalization left inconsistent [VERSION_MISMATCH]

v2 A65 and v3 Eq102 define IT=−Pi[T²]/4 and immediately assert IT=−6eta²Sigma. Their own main contraction is Pi[T²]=6eta²Sigma, which implies−3eta²Sigma/2 instead. Symmetry A70 retains that quarter factor despite main Eq12 switching to IT=−Pi[T²]. Hence this appendix is not a convention-matched derivation of the journal's main invariant coefficient.

## Recovery adjudication

Present: explicit supplementary action, an algebraic trace equation, exact multiplier enforcement, and a classical hard-limit condition. Missing: correct generic finite solution in the source, matched original couplings/eta, complete action variation, consistent arbitrary projective gauge action, and a derived low-energy tensor operator. Classification: **PARTIAL_DERIVATION with explicit CONTRADICTED_AS_WRITTEN finite-mass inference**, not recovery of the missing full C1/C3 proof. The same error is propagated v2→v3→Symmetry rather than resolved by an earlier version.
