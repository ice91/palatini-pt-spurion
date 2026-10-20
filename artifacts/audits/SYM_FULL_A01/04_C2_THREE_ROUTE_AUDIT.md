# C2 three-route audit

## Separate route outcomes

|Status key|Verdict|Meaning|
|---|---|---|
|ROD_STATUS|PROVED_WITHIN_STATED_DOMAIN for its determinant coefficient; CONDITIONAL for physical use of C1|A31–A33 algebra is correct with the specified traceless matrix and non-null normalization|
|CM_STATUS|CONTRADICTED_AS_WRITTEN|The actual rank-one metric shift gives a different Jacobian and quadratic coefficient|
|CS_PLUS_STATUS|BLOCKED_BY_MISSING_SOURCE for any additional nonzero shadow; CONTRADICTED_AS_WRITTEN for ordinary NY interpretation|Ordinary pure-trace Nieh–Yan and T∧T vanish; no additional nonzero CS+ operator or normalization is defined|
|THREE_ROUTE_EQUIVALENCE_STATUS|CONTRADICTED_AS_WRITTEN|ROD and CM already disagree at the claimed order; zero ordinary NY cannot supply the same nonzero bulk term|

PDF sources: pp.13–15 Eqs25–29, Proposition1; pp.33–35 A30–A41. Original program sources: `equivalence/{dbi_chain,closed_metric_chain,cspp_chain,coeff_extractor,order2_checker,__init__,flux_ratio}.py`, three coefficient JSON files, `notebooks/10_c2_symbolic_demo.py`, `scripts/fig_c2_coeff_compare.py` and `fig_flux_ratio.py`. All tests below use the published densities, not those JSON numbers.

## D1–D4 — actual determinant expansions

Work on a sign-fixed non-null gradient patch. Define the exact paper notation

```
sigma=sgn(Sigma_epsilon),  n_mu=v_mu/sqrt(|Sigma_epsilon|),  n²=sigma,
P^mu_nu=sigma n^mu n_nu,  P²=P, TrP=1,
Tcal=P−I/4,  TrTcal=0,  Tr(Tcal²)=3/4,
x=lambda tau,  a=(2/3)x.
```

Tcal has eigenvalues3/4,−1/4,−1/4,−1/4. For ROD, Eq26/A31 is explicitly

```
L_ROD/sqrt(-g) = sqrt(det(I+a Tcal))−1
               = sqrt[(1+x/2)(1−x/6)^3]−1
               = −x²/12 + O(x³).
```

This follows either from the exact eigenvalue determinant or `sqrt det(I+X)=1+TrX/2+(TrX)²/8−Tr(X²)/4+...`. With Eq25, `tau²=−(3/2)sigma IT`, its quadratic coefficient is `(lambda²/8)sigma IT`. This part is an independently verified L1 identity after imposing the stated C1 relation. It is not a physical derivation of C1 or of perturbative tensor K,G. Here delta² is read as the paper's quadratic coefficient convention, rather than twice that coefficient as a raw second derivative.

For CM, the original printed metric is instead

```
gtilde_mu_nu = g_mu_nu + (2/3)lambda tau n_mu n_nu.
```

The matrix determinant lemma gives

```
det(gtilde)/det(g) = 1 + (2/3)lambda tau n² = 1 + (2/3)sigma x,
L_CM/sqrt(-g) = sqrt[1+(2/3)sigma x]−1
             = sigma x/3 − x²/18 + O(x³).
```

Thus its second-order coefficient is `(lambda²/12)sigma IT`, **not** lambda²/8. The quadratic difference is

```
(L_CM−L_ROD)^(2)/sqrt(-g) = x²/36.
```

The coefficient discrepancy is independent of sigma, so removing a possible odd linear term does not fix it. Direct symbolic4x4 determinants check both timelike and spacelike n with rational arithmetic. A34 substitutes `Tcal` for the mixed rank-one matrix `n^mu n_nu`; those are different matrices. The mistake first occurs at the displayed determinant ratio immediately following the metric shift on p.34.

No compensation, subtraction, volume normalization or valid field transformation connecting these exact densities is supplied. A simple coefficient rescaling would itself change the declared route, and the audit does not perform one. A metric field redefinition of the *full* action would also require transforming EH and matter, not merely equating two volume elements. No such calculation is present.

At fixed C1, the difference is proportional to |d epsilon|²; for arbitrary metrics this is not a universally metric-independent boundary density. On a homogeneous timelike spurion background it has nonzero lapse/metric variation. Even though epsilon is not varied in A6, that metric variation remains. It cannot be dismissed as a harmless constant or an unspecified improvement.

## D5–D6 — ordinary Nieh–Yan and nonzero CS+

From `T^A=de^A+omega^A_B wedge e^B` and `R^A_B=domega^A_B+omega^A_C wedge omega^C_B`, the conventional identity is

```
d(e_A wedge T^A)=T_A wedge T^A−e_A wedge e_B wedge R^{AB}.
```

The main text p.14 rearranges this with the plus curvature term; Appendix C.3 instead writes the opposite rearrangement. This local sign mismatch does not affect the zero result below, and its precise correction is not used to infer nonzero physics.

For strictly pure trace the C1 torsion two-form is proportional to `e^A wedge v`, v=d epsilon. Therefore, pointwise,

```
e_A wedge T^A = 0,
T_A wedge T^A = 0,
NY=d(e_A wedge T^A)=0,
e_A wedge e_B wedge R^{AB}=0.
```

The first zero follows from symmetric contraction of `e_A wedge e^A`; the second also follows from the repeated one-form v. Both were verified by explicit antisymmetric-component contractions. A Hodge star on this zero4-form, or a scalar PT projector applied to zero, remains zero. Removing a genuine exact boundary term does not create a nonzero bulk scalar.

`T^A wedge star T_A` is a different, generally nonzero norm operator. Calling it a PT/Hodge shadow would require an explicit independently normalized definition and calculation; ordinary NY does not become that operator under a final star. Appendix C.3 provides only the identity and then asserts A35. There is no explicit nonzero density, no lambda-dependent coefficient derivation, and no source implementing one. Thus the nonzero CS+ coefficient is **SOURCE_MISSING / BLOCKED_BY_MISSING_SOURCE**, while its identification with the ordinary NY route is directly contradicted under pure trace. These are separate findings.

## D7–D8 — three independent routes versus a shared target

The program has two distinct, non-deriving APIs:

1. Figure path: each `*_chain.order2_raw` reads `I_T` and `Seps` from its JSON. All three shipped JSONs hold0.125 and0.0. `coeff_extractor.coeff_vectors` explicitly assumes these are already projected/IBP-reduced. No action, determinant, torsion, Levi–Civita tensor or metric variation is evaluated.
2. Original C2 test path: `equivalence/__init__.py` returns hardcoded `.125`; closed_metric adds an unlisted `total_derivative=1e−16`; cspp changes two chosen entries by ~1e−16. All `alpha` inputs are ignored. Hence tests at alpha=−3,0,4.2 do not independently probe coupling dependence.

`order2_checker` loops over thresholds and calls the same coefficient comparison. `ibp_tol` is unused by the extractors; the graph is not convergence of a physical reduction. The helper `reduce_to_IT(route=...)` checks the route label and then returns exactly the same formula for all three; the C2 notebook equality is built into that function. It also omits the sigma factor used in the publication, and its notebook takes Seps positive, masking that difference.

The independent temporary-config probe supplies7,11,13 in files **inside the audit temporary directory** and observes exactly those values returned. No original target/configuration was altered. This demonstrates coefficient provenance without inventing physical inputs or changing the theory. Values are arbitrary software sentinels only. Equality of the shipped target values is L4, not three L2 derivations.

## D9 and D12 — domain, shell and variation order

|Comparison domain|Finding|
|---|---|
|Before C1|No derivation of three unreduced route actions in terms of independent connection irreps; equivalence not established|
|After C1, at quadratic order|ROD and CM coefficients disagree; ordinary NY bulk is zero|
|Off shell for metric perturbations|No equal physical Hessians established; density differences have bulk variation|
|On shell|No supplied equations/solution proving the differences vanish; the audit does not impose extra on-shell conditions|
|Modulo valid boundary terms|No displayed valid improvement removes the determinant mismatch or supplies missing CS+ bulk|

If torsion is an algebraic auxiliary, substituting its actual full-action stationary solution before varying agrees with varying first by the auxiliary-field chain rule. Invertibility/differentiability and correct boundaries are required. C1 as printed is not shown to be that stationary solution, and the route weights can themselves affect the torsion equation. Using C1 first is consequently a conditional algebraic substitution, not a demonstrated equivalence of complete variational problems.

The ROD Taylor expansion is in lambda*tau (spurion/torsion amplitude), whereas a physical GW quadratic action is a second variation in h about a fixed background. These two uses of 'quadratic' cannot be silently identified. The source does not provide the missing full ADM variation to bridge them.

## D10 — improvements and flux ratios

A41 makes equal fluxes true by adopting an identical canonical current for every route. That convention does not show that the original actions have equal bulk terms. At a vanishing flux the ratio is0/0 and is not a defined invariant; finite nonzero equal fluxes require an actual current calculation.

There is also a direct counterexample to the printed canonical-current check. At a=1, A40 gives, for a single component h,

```
J0=(A_star/2)h dot h,  Ji=(A_star/2)h partial_i h,
partial_mu Jmu=(A_star/2)[dot h²+h ddot h+(grad h)²+h Laplacian h].
```

A38 at a=1 instead claims `(A_star/2)[dot h²−(grad h)²]`. For h=t²+x² these differ as explicit polynomials. Even changing a spatial sign leaves second-derivative terms unless additional equations or a complete density IBP identity is supplied. A quadratic Lorentzian kinetic action is not itself a pure divergence on arbitrary perturbations; its Euler–Lagrange derivative is nonzero. A36, A39 also mix covariant vectors and density factors/time conventions without the missing full derivation. No current formula from the source is presumed valid merely because it is called canonical.

The code `flux_ratio_FRW` returns `1+c R^(−sigma)` with default c=.5,sigma=1. It never constructs e,T,R or integrates a current on a ball. Its approach to1 is an illustrative programmed convergence curve. Even an independently computed ratio→1 would not alone establish equality of the bulk variations/principal symbols.

## D11 — sign and dimensions

For dimensionless epsilon and dimensionless metric in natural units, [v]=[T]=[tau]=mass, [Sigma]=[IT]=mass². The ROD determinant argument requires [lambda]=mass^−1, so [A_star]=mass^−2 and `[A_star IT]=1`. Eq26 as written is a volume-ratio density and needs an overall mass⁴ normalization to combine with a dimension-four EH density; this is not specified in the coefficient JSON. Assigning such a normalization would not fix the relative CM error. The action coefficients alpha1,alpha2,alpha3 then have mass² in Eq13. If a different epsilon dimension is intended, it must be stated consistently with t proportional to d epsilon.

Tau is signed before squaring. `tau²=9 eta² |Sigma|` and `sigma IT=−6 eta²|Sigma|` give a negative quadratic ROD contribution for real lambda, as the exact determinant confirms. The JSON stores only the coefficient +.125 multiplying IT and has no separate lambda or sign argument. The notebooks select positive Seps; they do not audit timelike sign compensation.

At Sigma=0 normalized n and sigma-based definitions are not ordinary smooth coordinates; conclusions here are on sign-fixed non-null domains. The paper's informal limiting-continuation remark does not by itself establish a unique differentiable action across arbitrary null loci.

No three-route physical equivalence survives the published densities. The ROD algebra remains valid as a narrower result; C2's numerical demonstrations remain reproducible demonstrations of chosen coefficients.
