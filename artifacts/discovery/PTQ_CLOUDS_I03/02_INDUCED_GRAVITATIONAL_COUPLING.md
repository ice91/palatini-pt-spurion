# I03 — induced curvature coefficient from the same scalar

**Calculation B completed:** with exactly the action, measure and proper-time UV cutoff of report 01,

\[
 \Gamma_m^\epsilon[g]\supset c_R^\epsilon\int d^4x\sqrt g R,
 \qquad c_R^\epsilon=-\frac{J(m,\epsilon)}{192\pi^2}.          \tag{16}
\]

Consequently S_ent^ε/A=−4πc_R^ε. This is a relation between the scalar's induced local source term and its regulated planar entropy. It is not a prediction of a finite measured Newton constant or an automatic metric equation.

## 1. Derive the smooth heat-kernel curvature coefficient

Use smooth complete Euclidean backgrounds, no physical boundary, weak compactly supported metric deformations and the positive operator D_g=−∇²+m². For the short-distance/local expansion,

\[
 K(s;x,x')\sim\frac{e^{-m^2s-\sigma(x,x')/(2s)}}{(4\pi s)^2}
   \Delta_V^{1/2}(x,x')[1+s a_1(x,x')+\cdots].              \tag{17}
\]

σ is half the squared geodesic distance. The heat-equation transport recursion with coincidence a_0=1 gives

\[
 [a_1]=[\Delta_V^{-1/2}\nabla^2\Delta_V^{1/2}].              \tag{18}
\]

In Riemann normal coordinates at x',

\[
 \Delta_V^{1/2}=1+\frac1{12}R_{\mu\nu}y^\mu y^\nu+O(y^3),
 \quad [\nabla^2\Delta_V^{1/2}]=R/6.
                                                               \tag{19}
\]

The last step is an explicit second derivative and contraction, verified for a general symmetric Ricci matrix. Therefore

\[
 \operatorname{Tr}e^{-sD_g}\sim
 \frac{e^{-m^2s}}{(4\pi s)^2}\int d^4x\sqrt g
       [1+sR/6+O(s^2R^2,s^2\nabla^2R)].                    \tag{20}
\]

The mass is factored out; it must not also be inserted as −m² inside a_1. The coefficient of s called a_1 here is denoted a_2 in the even-index convention of Vassilevich, *Heat kernel expansion: user's manual*, Eq.(4.27), [author text](https://arxiv.org/html/hep-th/0306138). The geometric transport identities are standard input; the contraction and induced integral are recomputed.

There is a separate spectral check without using the Van Vleck expansion. On an auxiliary Euclidean S² of radius a, the same scalar Laplacian has eigenvalues ℓ(ℓ+1)/a² and multiplicities 2ℓ+1. Setting t=s/a²,

\[
 \sum_{\ell\geq0}(2\ell+1)e^{-t\ell(\ell+1)}
       =t^{-1}+\frac13+O(t).                                \tag{21}
\]

The constant follows directly from Euler–Maclaurin: the integral is 1/t and the constant boundary correction is 1/2−2/12=1/3. Higher derivative corrections vanish at t=0. Since R=2/a², (21) agrees with the R/6 term in the trace. Independent spectral sums extrapolate to 0.33333333333315… . This checks the geometric coefficient on S²×R², with the mass factor removed; it is not a second physical field model/vacuum or a new gravity solution.

## 2. Perform the determinant integral and fix signs

Inserting (20) into (01:4) gives

\[
 \Gamma_R^\epsilon=-\frac1{2(4\pi)^2}\frac16
    \int_{\epsilon^2}^\infty\frac{ds}{s^2}e^{-m^2s}
          \int\sqrt g R,
                                                               \tag{22}
\]

which is precisely (16). c_R has dimension length⁻². This is the local curvature term extracted in the weak-curvature/derivative expansion; curvature-squared, nonlocal and higher terms are not declared absent. The UV expansion is controlled for curvature scales much longer than ε; a low-momentum local effective action for a massive field additionally requires external variations small compared with m. A heat-kernel asymptotic series is not a proof of all-background convergence or stable gravity.

Choose the usual Euclidean Einstein–Hilbert sign **only as the definition of how to compare coefficients**:

\[
 I_{g,E}\supset-\frac1{16\pi G_B}\int\sqrt g R.
 \quad \delta(1/G):=-16\pi c_R^\epsilon
             =\frac{J}{12\pi}>0.                          \tag{23}
\]

Under the corresponding Lorentzian continuation the local EH coefficient is positive. Thus the scalar supplies a positive induced inverse coupling in this convention. We have not assumed a preexisting value of G_B or postulated stationarity of I_g+Γ.

The independently calculated area term and curvature term satisfy

\[
 \boxed{\frac{S_{\rm ent}^{\epsilon}}{A}
 =-4\pi c_R^\epsilon=\frac14\delta(1/G)}.                  \tag{24}
\]

For mε≪1, c_R=−1/(192π²ε²)+mass terms and δ(1/G)=1/(12πε²)+mass terms. Both contain exactly the same J, including its logarithmic mass contribution. No normalization is changed to obtain (24).

## 3. Why the coefficient relation has geometric content

Compute the curvature of a smooth cone cap rather than insert its integrated value. Use angular period 2π and

\[
 ds_2^2=dr^2+F_n(r)^2d\theta^2,\qquad
 F_n(r)=nr+(1-n)a\tanh(r/a).
                                                               \tag{25}
\]

F_n'(0)=1, F_n'(∞)=n, and F_n=r+O(r³) at the smooth axis. Its scalar curvature satisfies R_2=−2F_n''/F_n, hence

\[
 \int\sqrt{g_2}R_2=-4\pi[F_n'(\infty)-F_n'(0)]
                  =4\pi(1-n).                             \tag{26}
\]

Multiply by the transverse area. The first replica derivative of c_R∫R is then −4πc_R A, giving exactly (24). This explains why the same short-distance field fluctuations can contribute to both quantities. It is a local conical-variation statement for the **same determinant**. It does not equate matter entanglement entropy to every possible geometric entropy term.

The comparison concerns the derivative at n=1. At finite deficit the smooth-curvature expansion and the sharp-cone limit need not commute; terms nonlinear in n−1 cannot be extrapolated from ∫R alone. Report 01's exact S_2/S_1=3/4 is an explicit counterexample to that extrapolation. We do not use a high-curvature cap to claim that a truncated EH action is valid there at arbitrary n.

## 4. Regulator dependence: what survives changing it?

For a common n-independent covariant proper-time profile w(s/ε²), replace both determinants by −(1/2)∫₀∞ds w(s/ε²)Tr K/s. Both sides of (24) contain

\[
 J_w=\int_0^\infty\frac{ds}{s^2}w(s/\epsilon^2)e^{-m^2s}.
                                                               \tag{27}
\]

Thus their relation survives in this shared class; their absolute values need not. For the smooth profile w(t)=1−e^{−t³}, which also regulates the volume term, the leading ε→0 coefficient is

\[
 J_w\sim\epsilon^{-2}\int_0^\infty\frac{dt}{t^2}(1-e^{-t^3})
 =\frac{\Gamma(2/3)}{\epsilon^2},\qquad
 \Gamma(2/3)=1.3541179394264\ldots .                         \tag{28}
\]

Integration by parts reduces the integral to 3∫₀∞t e^{−t³}dt. The hard step w=Θ(t−1) instead gives leading ε⁻². This is a targeted regulator control, not a different microscopic field or a new UV completion. With fixed m>0, (28) is the leading UV coefficient; the massless expression checks that limit. Numerical quadrature independently verifies it.

A shared regulator is essential to a literal numerical comparison. An n-dependent cutoff, a noncovariant regulator without the appropriate restoring counterterms, or unrelated cutoff calibrations need not preserve the displayed bare coefficients. Dimensional subtraction can remove power divergences; that is not a proof of zero physical entanglement or of a predicted zero Newton coupling. The number 1/(48π) belongs to the stated proper-time cutoff convention, not to all possible microscopic regulators.

## 5. Bare terms and a precise non-prediction of Newton's constant

Covariance permits independent local metric source terms, including a volume term, c_R,B∫R and curvature-squared terms. They have no φ dependence. Even though they are not specified as part of the matter action, a renormalized metric generating functional requires their normalization conditions. At the area/EH level,

\[
 \frac1{G_R}=\frac1{G_B(\epsilon)}+\frac{J}{12\pi},\qquad
 \frac{A}{4G_B(\epsilon)}+S_{\rm ent}^\epsilon
                  =\frac{A}{4G_R}.                        \tag{29}
\]

This is the leading-area renormalization identity, not a microscopic count of bare gravitational entropy or all higher-curvature entropy. For **any** chosen finite positive inverse coupling k, one may set

\[
 G_B(\epsilon)^{-1}=k-J/(12\pi),                            \tag{30}
\]

and obtain G_R⁻¹=k with exactly the same microscopic scalar. This proves the failure to determine a unique physical coupling from the matter determinant alone. Bare counterterms are not separately measured propagating-mode couplings; their cutoff-dependent signs do not by themselves diagnose a physical ghost.

More directly, adding b∫√gR to a background functional changes no normalized matter correlator at fixed g:

\[
 Z_b[g,j]=e^{-b\int\sqrt gR}Z_m[g,j],\qquad
 \frac{Z_b[g,j]}{Z_b[g,0]}=\frac{Z_m[g,j]}{Z_m[g,0]},
 \quad c_{R,\rm total}=c_R+b.                              \tag{31}
\]

The induced part is fixed in the declared scheme; the independent total source contact term is not fixed by those normalized matter data. The added term's conical derivative is geometric counterterm entropy, not a change of the normalized matter reduced state. Changing a renormalization scheme while keeping a measured G_R fixed is mere reallocation between bare and loop terms. Choosing different unconstrained finite gravitational normalizations is the genuine missing physical input. Equation (31) does not assert that physically measured gravity can be changed by notation.

Setting G_B⁻¹=0 would give G_ind=12π/J, approximately 12πε² for one scalar. That is a conditional “all induced at this cutoff” normalization, not a consequence of (2), and ε is not yet a physical microscopic scale. It supplies neither a regulator-independent finite Newton constant nor an observed value. For N identical scalars, δ(1/G) and S_ent both multiply by N; their matching survives, but species dependence has not disappeared.

## 6. The volume term and the missing dynamical postulate

The same determinant also yields

\[
 \Gamma_0^\epsilon=c_0^\epsilon\int\sqrt g,\qquad
 c_0^\epsilon=-\frac1{32\pi^2}
 \int_{\epsilon^2}^\infty\frac{ds}{s^3}e^{-m^2s}
 \sim-\frac1{64\pi^2\epsilon^4}.                            \tag{32}
\]

This is a scheme-dependent vacuum source term, not a predicted renormalized cosmological energy. If one later postulates a dynamical metric, flat-space stationarity would require an independent volume normalization cancelling the total vacuum term; positive δ(1/G) alone does not make Minkowski a solution. This is an additional gap, not an inconsistency of the external-background matter theory.

Diffeomorphism covariance of Γ gives a Ward identity: writing δΓ=(1/2)∫√g T_Γ^{μν}δg_μν and δg_μν=∇_μξ_ν+∇_νξ_μ, integration by parts gives ∇_μT_Γ^{μν}=0. It does **not** give δΓ/δg=0. The latter would require the extra physical postulate that g is a dynamical variable governed by stationarity or a gravitational path integral, plus its action/measure, renormalization and boundary/state data.

If that postulate is made, the leading local equation can have Einstein form after counterterms and a valid low-energy truncation. It is then conditional dynamics of the assumed metric theory. Since φ was integrated out, one must not absorb its local loop terms into G_R and simultaneously put its full unsplit stress functional on the other side: retain Γ_rem after subtraction, or use the full original functional consistently. Real-time causal backreaction also requires an in-in/state prescription; the Euclidean determinant is not a retarded response kernel.

No metric propagation, positive gravitational Hilbert space, full stability result or universal local Clausius equilibrium has been established by this source calculation. Evidence: `derive_induced_curvature.py` / `curvature_results.json` (18 exact checks), plus independent sphere-spectrum, cap, integral, regulator and normalization controls in `test_independent.py`.
