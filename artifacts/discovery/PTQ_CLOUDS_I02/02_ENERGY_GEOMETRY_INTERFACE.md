# Energy–geometry interface: what the calculation does and does not supply

**Decision:** the calculation gives a genuine QFT information–stress relation on a specified spacetime. It does not independently derive geometric dynamics. A standard local-horizon argument produces an Einstein-form equation only after separate geometric entropy and equilibrium premises are supplied. Those premises are not projection/modular theorems.

## 1. The actual bridge already obtained

In report 01 the fixed vacuum and wedge determine a modular flow; the coherent displacement then gives

\[
 D_R(\omega_\lambda\Vert\omega_0)
 =2\pi\int_0^\infty x\,\Delta\langle T_{00}\rangle dx
 =2\pi dE_{\rm cl}.                                        \tag{25}
\]

Each link has a type: access is an algebra, preparation is a state, modular flow is an algebra automorphism group, and T_μν is the stress operator of the action. The boost orbit/geometric weight x belongs to the supplied Minkowski geometry. Equation (25) relates information distinguishability to its boost-energy expectation. It is not a variation of the metric, an equation for curvature, or a determination of Newton's constant.

In the same state D_R>0 and Δ〈T_00〉>0, yet the prescribed metric remains flat. Rindler coordinates have

\[
 ds^2=\rho^2d\eta^2-d\rho^2,\qquad R_{\mu\nu}=0.             \tag{26}
\]

Direct Christoffel/Ricci computation verifies (26). Acceleration, modular temperature and nonzero information do not by themselves require curvature. This is a consistent nongravitational QFT, not a solution of an assumed sourced Einstein equation.

## 2. An important dimensional obstruction, computed explicitly

The actual parent is 1+1 dimensional. In two dimensions the Riemann symmetries imply

\[
 R_{abcd}=\tfrac R2(g_{ac}g_{bd}-g_{ad}g_{bc}),\quad
 R_{ab}=\tfrac R2g_{ab},\quad G_{ab}=0.                      \tag{27}
\]

An independent tangent-frame contraction checks this identity. The coordinate derivation uses a general local conformal metric g=e^{2c(t,x)}diag(1,−1):

\[
 R_{tt}=-c_{tt}+c_{xx},\quad R_{xx}=c_{tt}-c_{xx},\quad
 R=2e^{-2c}(-c_{tt}+c_{xx}),\quad G_{ab}\equiv0.             \tag{28}
\]

Every 2D Lorentzian metric is locally conformally flat, so this is not just a flat-coordinate test. Ordinary 2D Einstein–Hilbert bulk variation carries no local Einstein tensor. There is also no variable transverse-area density of the sort used in a higher-dimensional pencil of horizon generators; the usual Raychaudhuri screen dimension D−2 is zero.

At t=0 the actual flat-space coherent stress is

\[
 \Delta T_{ab}=\frac{\lambda^2f'^2}{2}
             \begin{pmatrix}1&0\\0&1\end{pmatrix},\qquad
 \eta^{ab}\Delta T_{ab}=0.                                 \tag{29}
\]

Imposing G_ab+Λg_ab=8πGΔT_ab without any further fields would force Λ=0 by the trace and then ΔT_ab=0, contradicting (29) wherever f'≠0. Thus importing an ordinary sourced Einstein equation into this 2D model does not produce a gravitational completion. Dilaton gravity, induced nonlocal actions, extra dimensions or other gravitational variables would be additional theories and were not introduced. No claim about all possible 2D gravity models follows from this restricted obstruction. Curved-background quantum anomalies were not computed from (28); that equation is a geometric identity, and (29) is the flat-space excitation stress.

## 3. A conditional geometric implication, separate from the fixed parent

The following is a **NEWLY EXECUTED REPRODUCTION OF A KNOWN CONDITIONAL ARGUMENT**, not a second QFT parent and not an I01 prediction. Let spacetime dimension n≥3 and add the premises below. They are all additional to report 01.

| Added premise | Content and independent role |
|---|---|
| G1 metric dynamics/domain | A smooth variable Lorentzian metric, Levi-Civita null congruences, arbitrarily small horizon patches at every point and in every null direction |
| G2 near-equilibrium limit | Expansion θ and shear vanish at a chosen cut; gradients/fluctuations are small enough for a leading local thermodynamic description |
| G3 area entropy | A geometric horizon entropy variation δS_h=αδA, with constant positive area density α; neither its microscopic origin nor value follows from the current algebra |
| G4 heat/temperature prescription | δQ is the matter boost-energy flux across the patch; the same local boost fixes T=κ/(2π); a suitable locally vacuum-like/Hadamard regime is required |
| G5 universal Clausius condition | δQ=TδS_h for **every** such near-equilibrium local horizon, not just one fixed wedge and not an assertion that matter relative entropy is area entropy |
| G6 conservation and geometric identities | ∇ᵃ〈T_ab〉=0 and the metric contracted Bianchi identity; expectation values have a consistent semiclassical renormalization and negligible fluctuations on the chosen scale |

No semiclassical Einstein equation is assumed in the following algebra. Instead G3 and especially G5 supply the missing physical coupling. They must not be advertised as consequences of positivity or the entanglement first law.

Let kᵃ be the affine horizon generator, λ_aff∈[−ℓ,0], and χᵃ=−κλ_aff kᵃ. Fix curvature sign by the Raychaudhuri convention

\[
 \theta'=-\frac{\theta^2}{n-2}-\sigma_{ab}\sigma^{ab}
                 -R_{ab}k^ak^b.
                                                               \tag{30}
\]

The twist vanishes for the horizon congruence. At the equilibrium cut θ(0)=σ(0)=0, so to leading order

\[
 \theta(\lambda_{\rm aff})=-\lambda_{\rm aff}R_{kk}
 +O(\lambda_{\rm aff}^2),\quad
 \delta A=-\int\lambda_{\rm aff}R_{kk}\,d\lambda_{\rm aff}dA,
 \quad \frac{\delta Q}{T}
 =-2\pi\int\lambda_{\rm aff}\langle T_{kk}\rangle d\lambda_{\rm aff}dA.
                                                               \tag{31}
\]

For a constant-curvature/stress infinitesimal pencil of initial area A_0, actual integration yields

\[
 \delta A=A_0\ell^2R_{kk}/2+O(\ell^3),\quad
 \delta Q/T=\pi A_0\ell^2\langle T_{kk}\rangle+O(\ell^3).
                                                               \tag{32}
\]

G3–G5 equate the leading coefficients, giving the conditional null equation

\[
 R_{kk}=\frac{2\pi}{\alpha}\langle T_{kk}\rangle.              \tag{33}
\]

Here κ cancels: arbitrary normalization of the local boost does not determine α. To derive the tensor equation rather than assume it, let C=2π/α and S_ab=R_ab−C〈T_ab〉. If S_ab kᵃkᵇ=0 for every null direction, then S_ab=F g_ab. Proof in an orthonormal frame: k=e_0±e_i forces S_0i=0 and S_ii=−S_00; k=e_0+(e_i+e_j)/√2 forces S_ij=0 for i≠j. An independent exact 3D null-direction solve verifies this algebra.

Taking the divergence gives ∂_bF=(1/2)∂_bR by G6. Thus F=R/2−Λ, with Λ an integration constant, and

\[
 \boxed{G_{ab}+\Lambda g_{ab}
              =\frac{2\pi}{\alpha}\langle T_{ab}\rangle}.
 \quad \alpha=\frac1{4G}\ \Longrightarrow\ \frac{2\pi}{\alpha}=8\pi G.
                                                               \tag{34}
\]

This reproduces the known local-horizon equation-of-state argument of Jacobson, *PRL*75,1260 (1995), Eqs.(1)–(6), [primary text](https://arxiv.org/html/gr-qc/9504004). We independently integrate (32), solve its coefficient and check the null-tensor algebra; no new gravitational discovery or determination of α, G or Λ is claimed. Equations (30)–(34) are leading-local conditional implications; exact field equations require demanding their infinitesimal limit at all points/directions under the stated equilibrium regime.

## 4. Where a false derivation would enter

Four distinct expressions must stay distinct:

\[
 \delta S_{\rm matter}=\delta\langle K_0\rangle
 \quad\hbox{(state variation, fixed reference/algebra)},
\]
\[
 D=\Delta\langle K_0\rangle-\Delta S_{\rm matter}
 \quad\hbox{(finite states, valid matrix/relative domain)},
\]
\[
 \delta S_h=\alpha\delta A
 \quad\hbox{(additional geometric entropy hypothesis)},
 \qquad \delta Q=T\delta S_h
 \quad\hbox{(additional equilibrium coupling)}.              \tag{35}
\]

In report 01 a local coherent unitary has unchanged regulated entanglement spectrum, yet finite D=ΔK>0. Thus identifying D with matter entropy increase would already fail this explicit state example. Identifying matter state variation with geometric area variation is also a separate step: one varies a state on a fixed region, the other varies the metric/horizon. Nothing in (20) of report 01 equates those directions or their entropies.

If one instead **assumes** semiclassical Einstein equations and integrates their null contraction through Raychaudhuri, equations (31)–(34) run in reverse and reproduce the area balance. That is a conditional reformulation of the law already supplied. D02's Einstein–Cartan action likewise contains an EH gravitational seed; importing its metric equation would not be an information-theoretic derivation and would change this sprint's parent. Neither route is used as new evidence for the historical Symmetry claims.

## 5. Premise removal and explanatory compression

| Remove this input | What remains |
|---|---|
| A geometric entropy/area law | QFT modular energy and relative entropy remain; no area variation or Newton coupling is fixed |
| Universal near-equilibrium Clausius condition | One can compare states on a fixed background, but cannot constrain its Ricci tensor through (33) |
| All-null/all-point requirement | At most one flux/focusing relation, insufficient to determine a tensor field equation |
| Stress conservation | The null equation does not integrate to the same Einstein form with constant Λ |
| Vacuum/wedge reference | Modular theory still exists for faithful states, but the simple local boost integral need not be its generator |
| PT, quaternionic labels, OBP | Every calculation in reports 01–02 still follows from ordinary current QFT and, for the template, G1–G6 |

There is a real mathematical interface: modular energy identifies the same stress/boost weight that can enter horizon focusing. There is no assumption-free bridge to dynamics here. The gravitational template adds physical premises rather than compressing them away. Its area coefficient, equilibrium justification, valid semiclassical state/backreaction regime and microscopic meaning of horizon entropy remain unexplained. A fixed wedge says nothing about evaporation or asymptotic radiation recovery.

Evidence: `evidence/derive_geometry_interface.py` / `geometry_results.json` (9 exact checks); independent curvature-tensor/null-direction/patch-integral checks in `test_independent.py`. The Ricci computations are actual calculations; the entropy/Clausius premises are assumptions, not “tests passed.”
