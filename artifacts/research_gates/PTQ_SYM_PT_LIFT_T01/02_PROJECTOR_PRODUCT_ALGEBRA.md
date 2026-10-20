# Gate 2 — projector product algebra

**Conditional algebra established; original Appendix A Eq.(A2) fails.** These results are generic Z2 algebra. They neither supply Symmetry's missing field map nor derive gravitational dynamics.

## Exact hypotheses and proof

Let A be an associative algebra over R (or a complex algebra regarded as real), with an additive real-linear map U:A→A satisfying U²=I and U(XY)=U(X)U(Y). U may be anti-complex-linear. It must preserve product order; an anti-automorphism such as quaternionic conjugation alone is a different hypothesis. Define P±=(I±U)/2. No positivity, inner product or causality is required.

    P+²=(I+2U+U²)/4=P+,
    P−²=(I−2U+U²)/4=P−,
    P+P−=P−P+=0,
    P++P−=I,  UP±=±P±.

These projector relations require real-linearity and involution, not multiplicativity. Multiplicativity is required for the following composite identities. Write X=X++X−, Y=Y++Y−. Keeping the original order of all products,

    P+(XY)=X+Y+ + X−Y−,
    P−(XY)=X+Y− + X−Y+.

The product identity remains valid in a noncommutative associative algebra. In particular, odd times odd is even, not discarded. P+ is multiplicative on the even subalgebra, but is not generally an algebra homomorphism on A. Equality P+(XY)=P+X P+Y for a particular pair holds precisely when X−Y−=0 (or its admitted equivalence class vanishes).

Symmetry p.6 Eq.(6), Theorem 1; p.28 A1; p.29 App.A.3 assert the relevant involution/automorphism. The displayed algebra checks their conditional consequences independently, without certifying the actual field/composite extension.

## Products, forms, densities and integrals

| Domain | Additional hypothesis needed | Consequence / limit |
|---|---|---|
| Scalar functions | Closed real algebra and a multiplicative U, e.g. a specified pullback/conjugation | Above formulas apply; anti-linearity is handled over R |
| Tensor products/contractions | Compatible tensor/bundle maps and transformed inverse metric/internal pairing | Same even/odd product logic only after the contraction map is equivariant |
| Wedge products | U maps each form degree appropriately and U(α∧β)=Uα∧Uβ in the same order | P+(α∧β)=α+∧β+ + α−∧β−. No Koszul sign is added because no factors are exchanged |
| Quaternionic-valued products | A specified order-preserving automorphism, or an explicitly different order-reversing rule | Cannot infer the above same-order product formula from quaternionic conjugation q→qbar alone; bar(XY)=barY barX |
| Scalar densities | Density weights tracked: Dens^w×Dens^v→Dens^(w+v) | Two weight-one densities do not produce another integrable weight-one density. Use homogeneous components of a weight-graded algebra, or scalars with an explicit invariant volume form |
| Top-degree forms | Wedge respects degree; degree sum must not exceed dimension | Two 4-forms in 4D wedge to zero; that is not the product of scalar coefficients used in A2 |
| Integrated real functionals | A well-defined real-linear integral on weight-one densities/top forms, invariant domain/measure and a compatible integrable pairing | Integrate the composite identity; the odd-odd term generally survives. Invariance kills odd integrands, not even products of odd integrands |
| Complex integrals with anti-linear U | Correct conjugation of the result and choice of real sector | For pullback/conjugation, ∫UX=overline(∫X) on invariant domains. Literal ∫X=∫UX for every complex X would fail already for constant i on positive finite volume; p.28's real-scalar restriction avoids that particular counterexample |

No quotient by boundaries is used to discard a nonzero bulk odd-odd product. Such a quotient would need an actual exact-density identity and matching variations, not a name.

## Independent odd-times-odd counterexamples

1. **A2 in the favorable scalar reading.** Take U f(x)=f(−x), real X=Y=x, fixed measure dx on [−1,1]. P+X=P+Y=0, P−X=P−Y=x, P+(XY)=x². Hence LHS(A2)=0 and RHS(A2)=2/3. This repeats the existing audit's mathematical counterexample; it is not a new independent physical experiment.
2. **Four-dimensional counterpart in the published coordinate involution.** On Minkowski D=[−1,1]^4 with fixed invariant volume, θx=−x and U=conjugation plus scalar pullback, X=Y=x¹ are odd and real. ∫D X² d⁴x=16/3 while ∫D(P+X)(P+Y)d⁴x=0. The volume domain is PT-invariant. This refutes the universal functional identity even before gravitational choices. If extra fall-offs/compact support are demanded, take a nonzero smooth real PT-even bump χ supported in the box and X=Y=x¹χ; the RHS is strictly positive and all boundary fluxes vanish. No numerical integration is needed.
3. **Wedge counterpart.** For real forms and ordinary θ pullback, α=dt and β=dx¹ are each odd, while α∧β is a nonzero even 2-form. Their separate even projections vanish. Wedge with a suitable even 2-form gives a nonzero even 4-form when an integrated example is wanted.
4. **Anti-linearity and reality.** In A=C with U=conjugation, X=Y=i gives P+X=0, P−X=i, P+(XY)=−1=(P−X)(P−Y). Replacing both P± by “take real part after projection” would destroy this identity: the modified odd part becomes zero and P++P− is no longer I on the complex domain. Any extra real-part map in Symmetry A1 needs its own domain; it must not be silently inserted into the above proof.

## What is wrong with original A2?

Symmetry p.28 Proposition A1 Eq.(A2) claims ∫Π[X]Π[Y]=∫Π[XY] for arbitrary scalar densities, calling it self-adjointness; p.30 App.A.6 uses it in C2. In the scalar-with-fixed-measure reading, the correct difference is

    ∫P+(XY) − ∫(P+X)(P+Y) = ∫(P−X)(P−Y).

PT invariance supplies no reason for the even right-hand side to vanish. For a real invariant L² pairing one instead has the ordinary self-adjointness statement <P+X,Y>=<X,P+Y>, and <P+X,P+Y>=<X,P+Y>. Those are different equations, stated here for comparison only; they do not replace or repair the published A2. On literal density×density the original expression has an additional weight issue.

## Exact calculations executed here

A standard-library free associative algebra calculation represented U(x)=X, U(X)=x, U(y)=Y, U(Y)=y. Polynomials have exact Fraction coefficients; multiplication concatenates words, and U replaces each letter without reversing word order. Projector identities were checked on x,y,xy and both composite identities were expanded exactly. The same bounded run checked Particles Eq.(12)–(13) 2×2 matrices and the noncommuting Schur/PT sign relations used in report 03. No project code, physical action or new dynamical block was executed/constructed.

```json
{
  "method": "Python standard-library fractions; free associative algebra; exact 2x2 integer matrices",
  "passed": 25,
  "results": {
    "x_U2": true,
    "x_Pplus2": true,
    "x_Pminus2": true,
    "x_orthogonal": true,
    "y_U2": true,
    "y_Pplus2": true,
    "y_Pminus2": true,
    "y_orthogonal": true,
    "xy_U2": true,
    "xy_Pplus2": true,
    "xy_Pminus2": true,
    "xy_orthogonal": true,
    "free_noncommutative_even_product": true,
    "free_noncommutative_odd_product": true,
    "A2_1D_counterexample": true,
    "A2_4D_counterexample": true,
    "Theta2": true,
    "Theta_W_Theta_minus_W": true,
    "P_W_P_zero": true,
    "Q_W_Q_zero": true,
    "P_W_Q_nonzero": true,
    "Q_W_P_nonzero": true,
    "W2_minus_I": true,
    "Schur_sign_minus_B_R_minusB": true,
    "PT_odd_B_even_R_even_correction": true
  }
}
```

Execution exit 0. Matrix/sign checks verify only the written finite algebra, not existence of a global differential inverse, PT lift or physical state. Prior audit evidence: `artifacts/audits/SYM_FULL_A01/03_C1_INDEPENDENT_AUDIT.md` §C1.11; `artifacts/research_gates/PTQ_SYM_ADMISSIBILITY_G01/01_PT_PROJECTOR_TYPECHECK.md`.
