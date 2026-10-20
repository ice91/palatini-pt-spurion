# Gate B — three-chain equivalence ladder

**ROD/CM: E1 on a sign-fixed, non-null, imposed pure-trace quadratic domain; E2 is contradicted for the written densities. CS+: E0 for an unspecified extra bulk operator. Ordinary NY is explicitly zero on that domain. No pair reaches E3/E4.** There is no independently established nonzero shared operator for all three original routes.

Primary: Symmetry pp.13–15 Eqs.(25)–(29), p.33 App.C notation/A30, p.34 A31–A35, p.35 A36–A41. Historical evolution: `artifacts/audits/SYM_HIST_P01/04_C2_C3_SOURCE_RECOVERY.md` H12–H14; full original audit: `artifacts/audits/SYM_FULL_A01/04_C2_THREE_ROUTE_AUDIT.md` D1–D12. The historical audit corrects the scope of the first audit's abstract traceless-matrix check; both are preserved.

Write v=dε, Σ=Π[v²], σ=sgnΣ, n=v/√|Σ|, x=λτ. Reading n²=σ itself requires that the real selected representative have v²=Σ. Work only on that stipulated real sign-fixed patch, Σ≠0, near x=0 on the square-root branch continuous from +1. No smooth continuation across a null locus is proved by the paper's continuation remark.

| Item | Original ROD | Original CM | Original CS+ |
|---|---|---|---|
| Input | Mixed endomorphism Tcal built from n,σ; torsion amplitude τ | Rank-one covariant metric shift | Coframe, torsion, curvature, ordinary NY 4-form |
| Exact written density | √−g [√det(1+(2/3)x Tcal)−1], Eq.(26)/A31 | gtilde=g+(2/3)x n⊗n; √−gtilde−√−g, §5.1.2/A34 | NY=d(e_A∧T^A), then ordinary Hodge/scalar PT projection of stated identity, §5.1.3/A35 |
| Actual PT step supplied | Scalar selection and asserted C1 reduction; no full determinant PT lift | Same scalar selection; no independently verified transform of shifted metric | A2 star commutation and A5 boundary assignment; asserted nonzero remainder |
| Assumptions | C1 imposed, real λ, non-null n; paper additionally claims Tcal traceless | Same C1/patch, local shifted metric retains Lorentzian signature near x=0 | Strict pure trace, admitted boundaries; no independently normalized nonzero extra density |
| Small parameter | λτ, not the metric perturbation h | λτ, not h | No calculable independent λ-dependent nonzero bulk expansion supplied |
| Scalar direction at x² | τ² ∝ |v|² after imposed trace lock | τ² ∝ |v|² after same lock | Ordinary NY: zero; extra Hodge norm: unspecified |
| Coefficient | Abstract traceless/spacelike −x²/12; printed timelike +x²/36 with linear 2x/3 | σx/3−x²/18 | Ordinary NY zero, asserted λ²/8 in A35 underived |
| Boundary/improvement | A33 allows J_ROD=0; A36–A41 choose common current | No valid improvement shown to remove nonzero quadratic bulk mismatch | A5 can discard an actual exact NY form, cannot manufacture a bulk norm |
| Variational meaning | Volume-density model; no complete reduced tensor Hessian derived | Metric-dependent volume difference; coefficient differences can affect metric equation | Zero strict-pure-trace NY has no such bulk variation; unspecified shadow has no defined variation |
| Unproved identification | Literal matrix ≡ claimed traceless matrix on timelike patches; C1 full-action solution; τ² identity normalization | Rank-one nn Jacobian ≡ traceless endomorphism Jacobian | Ordinary star/NY ≡ new nonzero norm density with correct λ normalization |

## Literal ROD and CM algebra

The **printed** p.33 matrix is Tcal=σ(nn−I/4), with σ outside the whole parenthesis. For n²=σ,

    tr Tcal=1−σ,
    tr(Tcal²)=5/4−σ/2.

Thus σ=+1 gives trace 0/norm 3/4; σ=−1 gives trace 2/norm 7/4. Do not replace it by σnn−I/4 on the timelike branch. For the printed matrix:

    ROD spacelike: √[(1+x/2)(1−x/6)^3]−1 = −x²/12+O(x³).
    ROD timelike:  √[(1+5x/6)(1+x/6)^3]−1 = 2x/3+x²/36+O(x³).
    CM either sign: √[1+(2/3)σx]−1 = σx/3−x²/18+O(x³).

CM follows from the determinant lemma, not from a traceless matrix. Even on the unproblematic spacelike branch the x² discrepancy is 1/36. Removing an odd linear term by an additional amplitude-parity convention would leave this discrepancy; no such convention is needed or invented for the check.

If one *conditionally uses the paper's* τ²=−(3/2)σI_T, the spacelike abstract ROD coefficient becomes λ²/8 and CM becomes λ²/12. But p.13 Eq.(25) also prints τ²=9η²|Σ| while Eq.(17) gives σI_T=−6η²|Σ|: these imply τ²=−(3/2)σI_T; the displayed form (3/2)(−σI_T) is consistent. The validity as a full-action C1 consequence remains unproved; main vs App.G invariant normalization also differs historically. Raw x coefficients above do not rely on that identification. No normalization is changed here.

The determinant expressions have no supplied overall mass-density normalization connecting them to EH; the relative mismatch persists with a common overall factor. A rescaling of only a route or a full metric field redefinition, including EH, sources and matter, would require an explicit sourced equivalence map. None is supplied.

## Exact check newly executed in this gate (limited algebra only)

Installed Python standard-library `fractions.Fraction` was used, without project imports or dependency installation. Multiply the four factors (1+(2/3)t_i x), retain coefficients d1,d2, and return sqrt coefficients d1/2 and d2/2−d1²/8. Printed eigenvalues are (3/4,−1/4,−1/4,−1/4) or (5/4,1/4,1/4,1/4); CM has determinant coefficients ((2/3)σ,0). This repeats known audit identities by a small exact calculation; it is **not a newly independent physical experiment** or route/ADM derivation.

```json
{
  "columns": [
    "linear x",
    "quadratic x\u00b2"
  ],
  "exact": {
    "ROD_spacelike": [
      "0",
      "-1/12"
    ],
    "ROD_timelike": [
      "2/3",
      "1/36"
    ],
    "CM_1": [
      "1/3",
      "-1/18"
    ],
    "CM_-1": [
      "-1/3",
      "-1/18"
    ]
  },
  "CM_minus_ROD_spacelike_x2": "1/36",
  "old_config_determinant": "29/80"
}
```

All asserted equalities passed; execution exit 0. No general field equations or principal symbols were computed.

## CS+ zero and the Hodge distinction

For strict pure trace T^A=c e^A∧v, symmetric metric contraction gives e_A∧T^A=0 and repeated v gives T_A∧T^A=0. Hence NY=d(e_A∧T^A)=0. The ordinary curvature term related by NY also vanishes in this specialization, regardless of the local sign mismatch between main p.14 and A35. Applying ordinary * or a genuine linear projector to zero gives zero.

T^A∧*T_A is a different operator, generally nonzero. Its possible nonzero value is not contradicted by the NY zero identity. But A35 supplies no separate normalized density defining that operator as the CS+ route, no independent λ-dependent coefficient and no original variation deriving it. This gate neither equates it to NY nor adds it to the action.

## Pairwise ladder and assumption dependence

| Pair | Maximum supported relation | E2/E3/E4 assessment |
|---|---|---|
| ROD–CM | E1: same scalar direction at x² after the common imposed trace/background reduction | E2 contradicted even at σ=+1; E3/E4 unverified, no valid full-action/physical-data map |
| ROD–CS+ | E0 for unspecified nonzero shadow; ordinary NY is comparable as zero but supplies no shared nonzero direction | Claimed nonzero equal coefficient contradicted under ordinary NY reading; no E2–E4 |
| CM–CS+ | Same E0 / zero distinction | Same failure |

E0 means not comparable under current definitions; E1 is a direction only; E2 equal bulk coefficients modulo valid boundaries; E3 equivalent reduced dynamics under explicit map; E4 same observables for matching preparations/sources. No larger label follows from the smaller one.

The paper's post-lock one-dimensional operator space (p.8 Eqs.(9)–(14), Lemma 2; p.12 Corollary 1) makes any *retained* nonzero quadratic scalar proportional to Σ. For the apparent all-route convergence this is **SELECTION_BY_ASSUMPTION**, not three independent derivations. ROD and CM supply a genuine bounded algebraic E1 once their inputs are fixed, while neither the completeness of the allowed space nor the original CS+ reduction is validated by that fact. Equality of configured coefficients and a common chosen current/flux are prior demonstrations, not physical equivalence proofs.

Gate B stop: preserve E1 for the supported pair; do not reconstruct densities, add counterterms or force E2/E3.
