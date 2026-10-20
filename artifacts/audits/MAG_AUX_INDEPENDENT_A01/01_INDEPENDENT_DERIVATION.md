# Independent action → curvature → Hessian → reduced action derivation

Verdicts are scoped to the actual trace/vector domain. All computations below were derived before comparison with the manuscript's proposed answers. Covector contractions use signature −+++; spectator fields and their first jets are fixed when taking the auxiliary Hessian. The common nonzero density factor √−g is suppressed.

## 1. Domain and conventions — PASS within the defined model

Source: pp. 3–4, (3.1)–(3.7), Assumptions 3.1–3.3; Appendix B, pp. 19–20.

The adopted connection derivative index is the first lower index. In particular

\[
R^\lambda{}_{\rho\mu\nu}=\partial_\mu\Gamma^\lambda{}_{\nu\rho}-\partial_\nu\Gamma^\lambda{}_{\mu\rho}
+\Gamma^\lambda{}_{\mu\sigma}\Gamma^\sigma{}_{\nu\rho}-\Gamma^\lambda{}_{\nu\sigma}\Gamma^\sigma{}_{\mu\rho},\quad
T_\mu=T^\lambda{}_{\lambda\mu}.
\]

Nonmetricity is \(Q_{\lambda\mu\nu}=-\nabla_\lambda g_{\mu\nu}\), with traces \(Q_\lambda=g^{\mu\nu}Q_{\lambda\mu\nu}\) and \(U_\nu:=\widetilde Q_\nu=g^{\mu\lambda}Q_{\lambda\mu\nu}\). We retain this exact torsion trace convention; changing to \(T^\lambda{}_{\mu\lambda}\) would reverse its sign.

Start from a general vector distortion

\[
N^\lambda{}_{\mu\nu}=\delta^\lambda_\mu a_\nu+\delta^\lambda_\nu b_\mu+g_{\mu\nu}c^\lambda.
\]

Direct evaluation of the definitions gives, separately in each covector component,

\[
\begin{pmatrix}T\\Q\\U\end{pmatrix}=
\begin{pmatrix}3&-3&0\\2&8&2\\5&2&5\end{pmatrix}
\begin{pmatrix}a\\b\\c\end{pmatrix},\qquad
\begin{pmatrix}a\\b\\c\end{pmatrix}=
\begin{pmatrix}1/3&5/36&-1/18\\0&5/36&-1/18\\-1/3&-7/36&5/18\end{pmatrix}
\begin{pmatrix}T\\Q\\U\end{pmatrix}.
\]

Thus (4.13) is recovered without using its coefficients as inputs. Axial torsion, tensor torsion, and traceless nonmetricity have not been varied or solved to zero. Their absence defines this 12-component connection domain, with four additional independent components of A. This establishes no consistent truncation of the unrestricted 64-component connection equations.

The action being varied is exactly

\[
S=\int\sqrt{-g}\,[f_1R(\Gamma)+f_2(T-A)^2+f_3\bar\nabla\!\cdot(T-A)+U(\phi)].
\]

The barred derivative is read as the Levi-Civita derivative, as required by (5.1), (5.5), and (5.6). The manuscript should define this explicitly adjacent to (3.7). Reading it as the full affine divergence changes the action and does not justify the stated integrations by parts.

## 2. Curvature identity — PASS

Source: p. 6, (5.1); Appendix C, pp. 20–21.

Contracting the curvature difference directly produces

\[
W^\mu=g^{\rho\nu}N^\mu{}_{\nu\rho}-g^{\rho\mu}N^\lambda{}_{\lambda\rho}
=-2T^\mu-Q^\mu+U^\mu,
\]

\[
\mathcal Q=g^{\rho\nu}(N^\lambda{}_{\lambda\kappa}N^\kappa{}_{\nu\rho}
-N^\lambda{}_{\nu\kappa}N^\kappa{}_{\lambda\rho})
=-\frac23T^2-\frac{11}{72}Q^2+\frac1{18}U^2-\frac23T\cdot Q+\frac23T\cdot U+\frac29Q\cdot U.
\]

Independent differentiation with arbitrary first jets of all three traces verifies the derivative term \(\bar\nabla_\mu W^\mu\). Hence \(R=R_{LC}+\bar\nabla_\mu W^\mu+\mathcal Q\). Normal coordinates are used only to evaluate a tensor identity at a point, not to assume a flat physical background.

Set \(p=df_1\), \(k=df_3\). The two actual integration-by-parts identities give

\[
L_{\rm raw}=f_1\mathcal Q+p\cdot(2T+Q-U)+f_2(T-A)^2-k\cdot(T-A).
\tag{Audit 1}
\]

This polynomial, derived from the stated action, is the only input to subsequent Hessian and stationarity calculations. The retained boundary current is exactly \(f_1W+f_3(T-A)\).

## 3. Arbitrary-one-form projective invariance — PASS

Source: pp. 4–6, (4.1)–(4.15); Appendix A, pp. 18–19.

For \(C^\lambda{}_{\mu\nu}=-\delta^\lambda_\nu\xi_\mu\), direct finite substitution into the curvature with arbitrary original connection gives

\[
R(\Gamma+C)^\lambda{}_{\rho\mu\nu}-R(\Gamma)^\lambda{}_{\rho\mu\nu}
=-\delta^\lambda_\rho(\partial_\mu\xi_\nu-\partial_\nu\xi_\mu).
\]

All mixed connection–C and quadratic C terms cancel. Its Ricci contraction is antisymmetric, so the scalar curvature is unchanged. The independently reconstructed trace map gives

\[
(\delta T,\delta Q,\delta U,\delta A)=(3,-8,-2,3)\xi.
\]

Both \(t=T-A\) and \(W\) are invariant. Consequently the original density, including its Levi-Civita divergence, is invariant for arbitrary smooth \(\xi\), including \(d\xi\ne0\). A is varied as an independent one-form, so no scalar-gradient integrability assertion enters.

For the manuscript's two-parameter ansatz \(C=a\delta^\lambda_\mu\xi_\nu+b\delta^\lambda_\nu\xi_\mu\), the derivative Ricci part is \(b\partial_\rho\xi_\nu+(-3a-b)\partial_\nu\xi_\rho\); its quadratic Ricci part is \(3a^2\xi_\rho\xi_\nu\). The specified Ricci and compensator targets select \(a=0,b=-1\) within that family. This is no uniqueness theorem about all possible gauge laws.

Metric compatibility is not assumed. Starting from a metric-compatible connection, this transformation generally produces \(\delta Q_{\lambda\mu\nu}=-2\xi_\lambda g_{\mu\nu}\); the MAG domain allows it and retains the shift in its trace sector.

## 4. Full varied Hessian — PASS

Source: pp. 7–10, (5.13)–(5.17), (6.1)–(6.12); Appendix F, p. 22.

Differentiate Audit 1 with respect to all 16 independent covector components, including A. In raw species order \((T,Q,U,A)\), the result is

\[
H_{\rm raw}^{sp}=
\begin{pmatrix}
-4f_1/3+2f_2&-2f_1/3&2f_1/3&-2f_2\\
-2f_1/3&-11f_1/36&2f_1/9&0\\
2f_1/3&2f_1/9&f_1/9&0\\
-2f_2&0&0&2f_2
\end{pmatrix},\quad J_{raw}=(2p-k,p,-p,k)^T.
\]

The true Hessian is \(\sqrt{-g}\,H_{raw}^{sp}\otimes g^{-1}\), not merely a 4×4 species matrix. Its raw gauge vector is \((3,-8,-2,3)\otimes\xi\). Differentiation verifies the exact off-shell Noether identity with these coefficients.

Introduce \(q=Q+8A/3\), \(u=U+2A/3\), \(t=T-A\), \(V=A\). This invertible, determinant-magnitude-one chart follows the trace shifts, not a target answer. Direct substitution of Audit 1 cancels every V and yields

\[
H_\perp=
\begin{pmatrix}
-11f_1/36&2f_1/9&-2f_1/3\\
2f_1/9&f_1/9&2f_1/3\\
-2f_1/3&2f_1/3&-4f_1/3+2f_2
\end{pmatrix},\quad J=(p,-p,2p-k)^T.
\]

The Palatini block P has \(\det P=-f_1^2/12\), and \(\det H_\perp=-f_1^2f_2/6\). Therefore the exact regular locus is \(f_1f_2\ne0\). On it the species rank is 3 and the full fibre rank is 12, with a four-dimensional kernel consisting exactly of arbitrary projective one-forms. Signs of f1 and f2 are irrelevant to this rank assertion.

## 5. Schur elimination and the stationary orbit — PASS

Source: pp. 8–11, (5.19)–(5.29), (6.13)–(6.15); Appendices D–G, pp. 21–23.

Solving the derivatives of Audit 1, rather than substituting the paper's solution, gives first

\[
q=4p/f_1-8t/3,\quad u=p/f_1-2t/3,
\qquad L_{\rm after\,(q,u)}=\frac{3p^2}{2f_1}+f_2t^2-k\cdot t.
\]

Thus \(S_t=2f_2\), \(J_{t,eff}=-k\), and \(t_\star=k/(2f_2)\). Setting \(r=p/f_1\), \(s=k/(2f_2)\), the entire raw stationary set is

\[
T=s+V,\quad Q=4r-\tfrac83(s+V),\quad U=r-\tfrac23(s+V),\quad A=V.
\]

All four raw Euler derivatives vanish, including the A equation. V is arbitrary and precisely parametrizes the gauge orbit. The solved distortion is

\[
N^\lambda{}_{\mu\nu}=\tfrac12(\delta^\lambda_\mu r_\nu+\delta^\lambda_\nu r_\mu-g_{\mu\nu}r^\lambda)
-\tfrac13\delta^\lambda_\nu(s_\mu+V_\mu).
\]

The raw and quotient calculations independently evaluate to

\[
L_{\rm aff,on}=\frac{3(df_1)^2}{2f_1}-\frac{(df_3)^2}{4f_2},\qquad
S_{red}=\int\sqrt{-g}\,[f_1R_{LC}+L_{aff,on}+U]+\text{retained boundary}.
\]

For coefficient functions of a single scalar, \(K_{eff}=3(f'_1)^2/(2f_1)-(f'_3)^2/(4f_2)\). This is an actual derived coefficient, not an assigned target. The Lorentz pairing lifts the one-species polynomial solve to all four components.

## 6. Singular branches, derived without dividing by zero — PASS

Source: pp. 13–15, Table 2; Appendix I, pp. 24–26. Substitute into the original horizontal linear system, not into the regular inverse. All compatibility conditions below are covector equations.

| Branch | Full fibre rank / nullity | Source compatibility | Compatible stationary data | Affine on-shell bulk |
|---|---|---|---|---|
| f1≠0, f2≠0 | 12 / 4 | automatic | unique horizontal solve | \(3p^2/(2f_1)-k^2/(4f_2)\) |
| f1=0, f2≠0 | 4 / 12 | p=0 | q,u arbitrary; t=k/(2f2); V gauge | \(-k^2/(4f_2)\) |
| f1≠0, f2=0 | 8 / 8 | k=0 | t arbitrary; q=4p/f1−8t/3, u=p/f1−2t/3; V gauge | \(3p^2/(2f_1)\) |
| f1=f2=0 | 0 / 16 | p=k=0 | q,u,t arbitrary; V gauge | 0 |

Extra horizontal null directions are not the specified projective gauge directions. This classification does not decide whether additional accidental symmetries arise for special constitutive functions on an entire singular open region. It does not permit a unique regular inverse to be extended across singular surfaces.

For a crossing with f1=0 and df1≠0 the auxiliary equations are incompatible. This is a pointwise obstruction, not a theorem about dynamically admissible trajectories, cosmology, or stability.
