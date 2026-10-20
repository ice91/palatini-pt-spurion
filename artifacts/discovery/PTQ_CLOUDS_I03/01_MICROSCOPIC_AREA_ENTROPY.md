# I03 — microscopic scalar area entropy

**Calculation A completed:** the vacuum of one minimally coupled real scalar gives a regulated planar entropy density

\[
 \frac{S_{\rm ent}^{\epsilon}}{A}
 =\frac{1}{48\pi}J(m,\epsilon),\qquad
 J=\int_{\epsilon^2}^\infty\frac{ds}{s^2}e^{-m^2s}.
                                                               \tag{1}
\]

The coefficient is derived below from angular/radial modes, not inserted as A/(4G). This is a newly executed reconstruction of known QFT, not a new entropy law. No Newton constant is an input to the field calculation.

## 1. One fixed microscopic parent and the assumption budget

Use signature (−,+,+,+), c=ℏ=k_B=1, one fixed mass m>0 and **minimal coupling ξ=0**:

\[
 S_L[\phi;g]=-\frac12\int d^4x\sqrt{-g}
       [g^{\mu\nu}\partial_\mu\phi\partial_\nu\phi+m^2\phi^2].
                                                               \tag{2}
\]

The metric is initially a nondynamical source. Calculation A uses Minkowski spacetime and its positive-frequency pure vacuum. The entangling surface is the plane Σ={t=0,x=0}, with right accessible region x>0 and causal domain x>|t|. A denotes its transverse area; an infinite plane has infinite total entropy. All expressions mean the area density, or the extensive term in an IR window with its edges sent away. No physical wall is introduced at Σ.

For Euclidean calculations set t=−iτ, with positive metric and curvature convention R>0 for a round sphere:

\[
 S_E=\frac12\int d^4x\sqrt g\,\phi D_g\phi,\quad
 D_g=-\nabla_g^2+m^2,\qquad
 Z_m[g]\ \hbox{formally}=(\det D_g)^{-1/2}.                 \tag{3}
\]

Use the standard L²(√g dx) Gaussian measure. On a complete smooth Euclidean background take the positive self-adjoint scalar Laplacian. On the replica cone use its **Friedrichs realization**, regular at the tip, with no δ-function interaction, alternative tip extension or imposed brick-wall boundary. At infinity use the usual decaying heat kernel. m>0 controls large-s infrared integrals. Weakly curved backgrounds in Calculation B are external Euclidean metric sources; the local UV coefficients can be continued back to Lorentzian signature. A global real Euclidean continuation of every Lorentzian metric is not asserted.

The **same explicit UV prescription** in A and B is

\[
 \boxed{\Gamma_m^\epsilon[g]
 :=-\frac12\int_{\epsilon^2}^{\infty}\frac{ds}{s}
                       \operatorname{Tr}e^{-sD_g}},\quad \epsilon>0.
                                                               \tag{4}
\]

This defines the proper-time regulated determinant scheme. Its normalization/local counterterm freedom is tracked in report 02; it is not silently declared a finite unique determinant. The cutoff ε has dimension length and s length². There is no chosen Planck length, fitted entropy coefficient, gravitational action or constitutive function in (2)–(4).

| Supplied ingredient | Why it is needed / what it does not supply |
|---|---|
| One free real scalar, fixed m and ξ=0 | Exact Gaussian integration and a definite curvature coupling; not a universal matter spectrum |
| Minkowski vacuum and planar cut | Euclidean gluing and product cone; no dynamical black-hole horizon or evaporation |
| Canonical Gaussian measure and positive operator domain | Fixes the determinant being computed; arbitrary tip interactions are excluded |
| Common covariant proper-time cutoff | Makes both coefficient comparisons meaningful; its physical microscopic origin is not derived |
| External metric sources | Allow computation of geometric response; no metric equation of motion follows by definition |
| Smooth weak-curvature/derivative expansion in B | Extracts the local R coefficient; not the entire effective action or its UV completion |

I01/I02's type-III warning is retained. An unregulated sharp continuum wedge is not assigned a finite density-matrix entropy. Equation (1) is a regulated replica quantity representing the matter vacuum area term; it is not relative entropy or a claim of a canonical finite-dimensional Hilbert factorization. A regulator for the determinant is not itself a constructed microscopic lattice theory.

## 2. Replica construction, with normalized state

Gluing n copies across the half-space cut gives M_n=C_n×R², where the angular period on C_n is 2πn. For integer n, the vacuum path integral yields

\[
 \log\operatorname{Tr}\rho_R^n=-\Gamma_n+n\Gamma_1,\qquad
 S_{\rm ent}^{\epsilon}
 =(n\partial_n-1)\Gamma_n^\epsilon\big|_{n=1}.              \tag{5}
\]

The positive cone spectral problem specifies the real-n continuation near n=1. The cutoff in (4) is independent of n; neither a local-temperature-dependent cutoff nor an n-dependent normalization is inserted. Bulk-volume divergences cancel from −Γ_n+nΓ_1, but the surface term need not cancel.

## 3. Derive the cone trace from modes

Angular eigenfunctions are e^{iℓθ/n}/√(2πn), ℓ∈Z. Radial modes are J_|ℓ|/n(kr) with spectral measure k dk. Integrating their heat factor gives the diagonal two-dimensional kernel

\[
 K_{C_n}(r,r;s)=\frac{e^{-z}}{4\pi ns}
        \sum_{\ell\in\mathbb Z}I_{|\ell|/n}(z),\quad z=\frac{r^2}{2s}.
                                                               \tag{6}
\]

The normalization follows from the radial Bessel orthogonality and ∫₀∞k dk e^{−sk²}J_ν(kr)²=e^{−z}I_ν(z)/(2s). At n=1, Σℓ I_|ℓ|(z)=e^z, recovering K=1/(4πs).

To perform the otherwise infinite radial trace introduce an **auxiliary radial IR/Abel regulator**, e^{−(p−1)z}, p=cosh u>1. This is not the UV cutoff ε. The Laplace identity

\[
 \int_0^\infty dz\,e^{-pz}I_\nu(z)
 =\frac{e^{-\nu u}}{\sinh u},\qquad p=\cosh u,\quad\nu\geq0
                                                               \tag{7}
\]

and the angular geometric sum give

\[
 T_n(u)=\frac{1}{2\sinh u}\sum_{\ell\in\mathbb Z}e^{-|\ell|u/n}
 =\frac{\coth[u/(2n)]}{2\sinh u},\quad
 T_1(u)=\frac1{2(\cosh u-1)}.                               \tag{8}
\]

Subtract the same regulated extensive flat term **before** removing u:

\[
 C(n)=\lim_{u\to0^+}[T_n(u)-nT_1(u)].                       \tag{9}
\]

Using coth[u/(2n)]=2n/u+u/(6n)+O(u³), the 1/u² pieces cancel, leaving

\[
 \boxed{C(n)=\frac1{12}(n^{-1}-n)}.                         \tag{10}
\]

This is the surface trace defect, calculated without G. Independent numerical Laplace integrations at noninteger Bessel orders verify (7). At n=1/q, an independent orbifold image calculation gives

\[
 C(1/q)=\frac1q\sum_{k=1}^{q-1}\frac1{4\sin^2(\pi k/q)}
       =\frac{q^2-1}{12q}.                                \tag{11}
\]

Here each image follows from directly integrating (4πs)⁻¹ exp[−|X−R_kX|²/(4s)] over the plane; the trigonometric sum follows from differentiating the sine product formula. Discrete image checks are independent checks, not a substitute for the real-n continuation in (8)–(10).

## 4. The area factor and entropy coefficient

The transverse directions have the ordinary flat Gaussian trace

\[
 \operatorname{Tr}_{\Sigma}e^{s\partial_\perp^2}
 =A\int\frac{d^2p_\perp}{(2\pi)^2}e^{-sp_\perp^2}
 =\frac{A}{4\pi s}.                                       \tag{12}
\]

Thus **the area factor comes from the number of transverse short-distance modes** multiplying a conical defect in the two normal directions. It was not an entropy-area axiom. Since the mass commutes with the Laplacian,

\[
 \operatorname{Tr}K_n(s)-n\operatorname{Tr}K_1(s)
 =e^{-m^2s}\frac{A}{4\pi s}C(n),\quad
 \Gamma_n^\epsilon-n\Gamma_1^\epsilon
 =-\frac{A}{8\pi}C(n)J(m,\epsilon).                        \tag{13}
\]

Equations (5),(10),(13) give C'(1)=−1/6 and therefore (1), including its sign and numerical coefficient.

The integral can also be evaluated, not just power-counted:

\[
 J=\frac{e^{-m^2\epsilon^2}}{\epsilon^2}
       -m^2 E_1(m^2\epsilon^2)
 =\frac1{\epsilon^2}
  +m^2[\log(m^2\epsilon^2)+\gamma_E-1]
  -\frac12m^4\epsilon^2+O(m^6\epsilon^4).                  \tag{14}
\]

J>0 and ∂J/∂m²=−E_1(m²ε²)<0. The leading UV area term is A/(48πε²). Its massless limit is a consistency limit, not a second microscopic model. The exact integral controls the mass dependence in this regulator. No finite absolute sharp-cut entropy survives taking ε→0.

For N identical independent copies, log Z and the replica defect add, giving S_N=N S_1. The actual parent is one scalar; this arithmetic species control does not assume that an arbitrary spin/gauge spectrum has the same coefficient.

## 5. A useful finite-replica counterexample

The same calculation supplies the regulated Rényi area term without a new model:

\[
 S_n^\epsilon=\frac{\Gamma_n-n\Gamma_1}{n-1}
 =\frac{AJ}{96\pi}(1+n^{-1}),\quad
 \frac{S_2}{S_1}=\frac34.                                  \tag{15}
\]

An Einstein–Hilbert term evaluated only with ∫R=4π(1−n)A would instead yield a constant Rényi contribution. The matching to that term in report 02 is a **first conical derivative at n=1**, not an equality of the full replica partition function or entanglement spectrum. Indeed C(2)=−1/8, whereas its curvature-linear approximation (1−2)/6 is −1/6. Nonlinear conical effects must not be discarded to force a stronger identity.

## 6. Evidence and prior-work boundary

`evidence/derive_area_entropy.py` computes (8)–(15) symbolically and performs exact image checks: 14 checks pass. `test_independent.py` reconstructs the Bessel integrals, radial image integrals, small-u limit and replica derivative without importing that script or its outputs. One initial half-angle simplification assertion failed; the true identity was normalized to exponentials and then passed. `area_attempt1.json` preserves the issue; no physical formula, domain or model was changed.

For comparison with established literature: Solodukhin, *Entanglement entropy of black holes*, §§2.8–2.10, Eqs.(20),(27)–(31), [author text](https://arxiv.org/html/1104.3712). Our mode-sum/Abel derivation is executed here; the known area law is not relabeled a novel discovery. I01/I02's interpretation of state restriction and continuum divergences remains intact. No previous source/audit was revised.
