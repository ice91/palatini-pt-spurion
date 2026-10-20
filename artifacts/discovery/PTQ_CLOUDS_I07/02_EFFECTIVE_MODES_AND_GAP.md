# I07 — effective constraint directions and the gap obstruction

**Result:** the coarse variable has nonsimple components outside I06's nineteen-constraint surface. The calculation identifies algebraic residual directions, not extra physical degrees of freedom. The conditional I06 scalar pole and Mβ²=4/(3λ) remain valid there, but are not obtained from the I07 parent/map.

## 1. Exact source of the constraint defect

Let \(\tilde b_i\) be the already pulled-back and internally transported microscopic matrices in the common reference/root frame, weights wᵢ summing to one, and \(\bar b=\sum_iw_i\tilde b_i\). Define δbᵢ=\(\tilde b_i-\bar b\). An exact algebraic identity is

\[
 \sum_iw_i\tilde b_iW\tilde b_i^T
  =\bar bW\bar b^T+\sum_iw_i\delta b_iW\delta b_i^T.        \tag{15}
\]

For hard-simple data on the same gravitational branch, each term on the left is vᵢW. Write \(\mathcal P_{19}(H)=H-\operatorname{tr}(WH)W/6-\operatorname{tr}(DH)D/6\). Then

\[
 \boxed{C_{19}(\bar b)
  =-\mathcal P_{19}\!\left(\sum_iw_i\delta b_iW\delta b_i^T\right).}
                                                               \tag{16}
\]

For the same two-cell map this is

\[
 C_{19}(\bar b)=-w(1-w)\mathcal P_{19}
    [(\tilde b_2-\tilde b_1)W(\tilde b_2-\tilde b_1)^T].    \tag{17}
\]

Thus the mean fails to carry the microscopic simplicity relation because its second geometric moments have been discarded. W has signature (3,3); this moment expression is **not** a positive statistical covariance or a positive effective potential. Even an ordinary positive variance would not supply a Lorentzian kinetic operator or a propagator pole.

The extra closure condition needed for I06 would be precisely that the projected moment in (16) vanish. It is neither entailed by hard microscopic simplicity nor by internal Lorentz covariance: the exact on-shell witness in report 01 violates it. The hard multipliers constrain each microscopic B, rather than forcing this equation on its linear mean. None of the stipulated real structure, orientation, branch or gauge symmetries protects all nineteen directions after this averaging.

## 2. More than one nonscalar residual direction

The failure is not limited to one isolated choice of large field variation. Within the same action and the same map, take instead e₂=diag(1+u,1+z,1,1), e₁=I. For positive coframe factors,

\[
 t_1-t_2=w(1-w)uz.                                       \tag{18}
\]

The C19 residual is second order in small coframe changes; its first derivatives at u=z=0 vanish while its mixed second derivative does not. A linearized average can therefore conceal the nonlinear nonclosure. Generic small nonzero u,z already suffice. These coframes also admit the smooth, exactly flat construction of report 01 with correspondingly smaller positive step amplitudes.

For a lower bound on independent directions, retain w=1/2 and use two admissible microscopic data sets with e₂=diag(2,3,1,1) and e₂′=diag(2,1,3,1). The residual vectors with entries ((01,23),(02,13),(03,12)) are respectively

\[
 c=(-1/3,-1/6,+1/6),\qquad c'=(+1/6,+1/3,+1/6).            \tag{19}
\]

The first two components have determinant −1/12. These are at least two independent C19 residual directions, while Q=0 for each. This rank statement concerns the span of coarse **constraint defects** over permitted microscopic inputs, not two new particle polarizations. We have not calculated the full image dimension, a nonlinear effective Hamiltonian or a generic physical degree count.

Because the residual is a polynomial in coframe entries and is nonzero at (19), its zero set is a proper algebraic subset in the corresponding finite-dimensional two-cell data space. Nonclosure persists on an open neighborhood of these regular data. This establishes generic nonclosure for this algebraic family; it does not assert that every solution ensemble, every state or every possible coarse map samples these directions equally.

## 3. Why there is no I07 scalar mass calculation

The comparison benchmark is an **established earlier conditional model result**:

\[
 S_{I06,red}=\frac1{2\kappa}\int\sqrt{-g}
 \left[R-\frac{3}{2(1+\beta^2)}(\partial\beta)^2
       -\frac{2\beta^2}{\lambda(1-\beta^2)}\right],
 \quad M_\beta^2=\frac4{3\lambda}.                        \tag{20}
\]

Source: `artifacts/discovery/PTQ_CLOUDS_I06/02_RELAXED_CONSTRAINT_PHYSICAL_TEST.md`, Eqs.(17)–(23). That action followed from a deliberately rank-one multiplier relaxation **with C19 still exactly imposed**. λ was a free input of that different model. It is absent from the present hard microscopic action (1).

Substituting a mean B outside C19 into (20) is not a valid derivation or field redefinition. Setting β=Q/(some volume) would miss (19), and adding a C19 projection would change the stated mechanism. No such repair is made.

| Requested spectral quantity | I07 status and reason |
|---|---|
| Single effective β coordinate | Not obtained: the actual image includes C19 violations with Q=0 |
| Quadratic β effective action | Not formed: no closed β sector from this map |
| Kinetic coefficient, residue and physical pole | Not established; a moment defect is not any of them |
| Positive microscopic mass gap | Not calculated, and not supported by this closure test |
| λ from microscopic data | No relation derived; I06's λ remains free |
| Dependence of a physical gap on ℓ | Undetermined; ℓ labels the prescribed resolution window, not a derived particle mass |

The witness is a pure coordinate description of flat microscopic geometry, so a nonzero algebraic coarse residual by itself does not even require a new microscopic excitation. It is particularly unsuitable evidence for a physical massive scalar. This does not prove that every possible coarse-grained hard theory lacks scalar composite excitations; such a universal spectral claim has not been tested.

## 4. Causality, probes and energy: what is not inferred

I06 derived a positive scalar kinetic term, physical scalar stress energy, and Maxwell direct blindness **after** solving its own C19 field equations. Those results cannot be transferred to this nonclosed coarse variable. In I07:

- No Maxwell source/detector law, reconstructed effective metric or effective constraint Hamiltonian has been added after the closure failure.
- No mode is designated gauge, heavy, nondynamical, dark matter or unobservable merely because the coarse map discards information.
- There is no claim that I06's β lacks energy; the statement is that I07 has not derived that β.
- The exact microscopic witness is regular, torsion-free and flat. This verifies microscopic consistency, not the positivity or causal cone of a hypothetical effective nonsimple action.

A local effective description could require additional moment fields, a different operational variable or a dynamically justified suppression mechanism. These are missing inputs, not additions authorized automatically by a failed closure test. The first early-stop condition already applies; no further spectral calculation is attempted.

## 5. Independent computation and adversarial scope

`evidence/derive_closure.py` executes 30 exact SymPy checks: microscopic constraints, weight polynomial, determinant, C19 and its invariant, moment identity, lower bound on residual span, quadratic onset, independent cell/root Lorentz-frame transport, and the connected coframe's dB=0/hard algebra. The remaining flat-parent equations F=φB=0 and εφ=0 follow analytically from A=φ=0; no tautological matrix equality is counted as an independent test.

`evidence/check_independent.py` executes 21 checks using only Fraction arithmetic and an independent dictionary exterior algebra. It imports neither the first script nor its outputs. It checks six rational weights including both endpoints, exact residual entries and the two-direction minor. The displayed analytic equations, not assertion counts, establish the theorem. Neither program performs a path integral or derives an effective particle spectrum.

The elementary two-coframe sum itself has a precedent in I05 and is acknowledged rather than counted as new evidence. These newly executed computations establish the stronger, map-specific C19 obstruction and its on-shell realization. There was no physical model revision or second mechanism; the second calculation cycle was independent falsification/verification of the first.
