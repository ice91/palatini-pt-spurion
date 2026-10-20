# I04 — a causal shear-stress response from one fixed scalar

**Derived matter response, not derived gravity.** A Wick calculation in the specified vacuum gives

\[
 \rho_{xy}(s)=\frac{\theta(s-4m^2)}{1920\pi^2}
       s^2\left(1-\frac{4m^2}{s}\right)^{5/2}.                 \tag{1}
\]

The positive two-particle continuum fixes an absorptive and nonlocal retarded response. It leaves local subtraction data undetermined. This is a newly executed calculation of established QFT, not new scalar physics or a PTQ theorem.

## 1. Fixed parent, domain and measured quantity

Use exactly I03's real scalar, m>0, ξ=0, signature (−,+,+,+), ℏ=c=1:

\[
 S_m=-\frac12\int\sqrt{-g}\,[g^{\mu\nu}\partial_\mu\phi\partial_\nu\phi+m^2\phi^2],
 \quad (\Box_g-m^2)\phi=0,
 \quad T_{\mu\nu}=\partial_\mu\phi\partial_\nu\phi
 -\frac12g_{\mu\nu}[(\partial\phi)^2+m^2\phi^2].             \tag{2}
\]

Here δS_m=(1/2)∫√−g T^{μν}δg_μν. There is no independent affine connection, compensator, improvement ξ, new matter field or constitutive function. The metric is an **external source** until report 02 explicitly changes its physical status by assumption. Flat fields satisfy [φ(t,x),π(t,y)]=iδ³(x−y), [a_p,a†_p′]=(2π)³2E_p δ³(p−p′), E_p=√(p²+m²). The state is the Minkowski positive-energy vacuum at t→−∞, evolved with the supplied perturbation. No initially excited matter or metric ensemble is hidden in it.

Take a smooth real weak source g=η+h, switched on after vacuum preparation. For the primary channel choose h_xy=h_yx=h(t,z), other components zero, with Fourier convention e^{−iωt+ikz}; q=(ω,0,0,k). The source couples as

\[
 S_{\rm int}^{(1)}=\int d^4x\,h O,\quad O=T_{xy}=\partial_x\phi\partial_y\phi.
                                                               \tag{3}
\]

There is no remaining factor 1/2: the xy and yx entries both occur. This is one transverse traceless spatial polarization, not a nonlocal spatial-TT projection on arbitrary data. For this q, δh_xy=iq_xξ_y+iq_yξ_x=0, so it is a physical tensor amplitude under linear gauge transformations of this momentum. Its linearized curvature is nonzero for generic h. The physical observable is the change of the calibrated shear stress, smeared with smooth test functions; pointwise products below mean renormalized distributions.

The plane-wave sector is understood per transverse area (or as the large transverse-window limit). Initially arbitrary smooth compact metric test sources define the local response; restriction to q_x=q_y=0 is made afterward. Thus the ideal infinite plane is not a compact experimental apparatus. Retarded support in this reduced channel means t≥0 and |z|≤t. No artificial finite-frequency cutoff is used to assert exact causality.

## 2. In-in contour and the sign of the susceptibility

Use the closed-time-path functional

\[
 Z_{\rm CTP}[g_+,g_-]=\langle0|U[g_-]^\dagger U[g_+]|0\rangle,
 \quad W_{\rm CTP}=-i\log Z_{\rm CTP}.                         \tag{4}
\]

The contour goes from the past vacuum to the future and back, with fields matched at the future end. Vary on the + branch before setting g_+=g_-=g. Although Z_CTP[g,g]=1, its independent-branch variation is not zero. At first order H_int=−∫hO. Directly expanding the interaction-picture unitary gives

\[
 \delta\langle O(x)\rangle=\int d^4y\,
 \underbrace{i\theta(x^0-y^0)\langle[O(x),O(y)]\rangle_0}_{\chi_R(x-y)}h(y)
 +L_{\rm ct}[h](x).                                           \tag{5}
\]

L_ct includes the explicit metric variation of the stress operator (the seagull), density/index conversion, and covariant source counterterms. It is not set to zero by writing a flat normal-ordered O. For gravity feedback the generalized force is δS_m/δh=√−g T^{xy}; its nonlocal flat response equals (5), while its contact bookkeeping is obtained by varying this full expression. Renormalized flat stress may be normalized to zero, but that is a volume normalization choice, not a prediction of vacuum gravity.

The seagull can be exhibited directly. Put C=(∂φ)²+m²φ². Expanding the **same** metric-coupled Lagrangian gives

\[
 \mathcal L_m=\mathcal L_0+h\partial_x\phi\partial_y\phi
 +\frac{h^2}{2}\left[\frac C2-(\partial_x\phi)^2-(\partial_y\phi)^2\right]+O(h^3).
                                                               \tag{5a}
\]

Thus L_ct contains δ⁴(x−y) times the expectation of the square bracket, before counterterms. For the lower-index T_xy itself its explicit operator variation is instead −C/2; the difference is index conversion. In covariant dimensional regularization,

\[
 B_d=\langle\phi^2\rangle_{\rm reg}
 =\frac{\mu^{4-d}m^{d-2}}{(4\pi)^{d/2}}\Gamma(1-d/2),\quad
 \langle\partial_\mu\phi\partial_\nu\phi\rangle_{\rm reg}
 =-\eta_{\mu\nu}\frac{m^2B_d}{d}.                             \tag{5b}
\]

The second equality follows from Lorentz invariance and the coincident free equation with scaleless contact integrals set to zero in this regulator. The seagull expectation is 2m²B_d/d, a divergent **local** coefficient. Keeping d until subtraction retains its evanescent finite part. It combines with the coincident bubble and covariant counterterms; it is not another nonlocal cut. The finite a_i in (12) remain explicit normalization data. This gives the channel's actual contact operator, rather than omitting it.

Our sign convention is explicit: for an oscillator iθ[q(t),q(0)]=θ sin(Ωt)/Ω, whose transform is [Ω²−(ω+i0)²]⁻¹. It has positive imaginary part at positive frequency. Authors defining Π with an extra minus sign must be translated, not copied into (5). A Feynman/in-out time-ordered kernel is not used here. Replacing (ω+i0)² by a prescription with the same sign of the imaginary invariant at both signs of ω would give the wrong retarded kernel.

## 3. On-shell matrix elements and the two-particle cut

With relativistically normalized identical-particle states, the flat stress vertex is

\[
 M_{\mu\nu}(p,p')=\langle0|T_{\mu\nu}(0)|p,p'\rangle
 =-p_\mu p'_\nu-p_\nu p'_\mu+\eta_{\mu\nu}(p\cdot p'-m^2).
                                                               \tag{6}
\]

Since p²=p′²=−m², q=p+p′ obeys

\[
 q^\mu M_{\mu\nu}=-(p\cdot p'-m^2)q_\nu
                         +(p\cdot p'-m^2)q_\nu=0.              \tag{7}
\]

This checks the conservation of the spectral discontinuity without guessing a transverse polarization function. The full response Ward identity also has coincident/tadpole terms: varying ∇_μ〈T^{μν}〉=0 gives ∂_μδ〈T^{μν}〉 plus δΓ·〈T〉 and index/density variations. Covariant regularization and the contact terms in (5) implement that identity. Only after a zero total flat tadpole is imposed is the full metric Hessian transverse with no background-tadpole remainder. A commutator-only calculation is insufficient to fix its contacts. The spin-zero channel is not calculated here; it cannot be inferred absent from the theory because h_xy does not excite it.

For q future timelike, let s=ω²−k², β=√(1−4m²/s). In the center-of-mass frame E=√s/2, |p|=√s β/2, p′=−p, and M_xy=2p_xp_y. The two-particle phase space and angular integral are computed as

\[
 d\Phi_2=\frac{|p|}{16\pi^2\sqrt s}d\Omega,
 \quad\int d\Phi_2=\frac{\beta}{8\pi},\quad
 \langle p_x^2p_y^2\rangle_\Omega
 =\frac{p^4}{4\pi}\int_0^\pi\sin^5\theta\,d\theta
                     \int_0^{2\pi}\cos^2\varphi\sin^2\varphi\,d\varphi
 =\frac{p^4}{15}.                                             \tag{8}
\]

Thus the connected Wightman transform, including 1/2! completeness, is

\[
 \widetilde W_{xy}(q)=\frac12\int d\Phi_2\,|M_{xy}|^2
 =\theta(\omega)\theta(s-4m^2)\frac{s^2\beta^5}{960\pi}
 =2\pi\theta(\omega)\rho_{xy}(s).                             \tag{9}
\]

The support and β⁵ threshold follow from the two-body delta function and shear matrix element; they were not inserted into a desired gravitational kernel. A boost along z leaves the xy component unchanged, so (9) applies to the declared q. Inserting states in (5) gives the formal dispersion integral ∫dσ ρ_xy(σ)/(σ−z), where

\[
 z=(\omega+i0)^2-k^2.                                         \tag{10}
\]

Its unrenormalized real part diverges; using it without subtraction would be a mistake.

## 4. Explicit renormalization prescription and exact finite integral

Define the covariant extension of the CTP stress distributions with dimensional regularization (d=4−2δ), local generally covariant counterterms, and **three finite Taylor subtraction conditions at z=0** in this channel. The convergent dispersive remainder below is an equivalent explicit definition of its nonlocal part. The real constants a_0,a_1,a_2 specify the remaining local normalization; they are not secretly zero physical gravitational parameters. A common covariant regulator translates into this prescription by local matching. In particular I03's finite-ε Euclidean proper-time determinant is not declared a causal Lorentzian cutoff kernel; report 02 translates its local R normalization only. Removing the UV regulator after counterterms is essential.

The elementary identity

\[
 \frac1{\sigma-z}-\frac1\sigma-\frac z{\sigma^2}-\frac{z^2}{\sigma^3}
 =\frac{z^3}{\sigma^3(\sigma-z)}                               \tag{11}
\]

gives the complete shear response up to its covariant contact completion:

\[
 \chi_R^{\rm ren}(z)=a_0+a_1z+a_2z^2+\widehat\chi_R(z),
 \qquad
 \boxed{\widehat\chi_R(z)=z^3\int_{4m^2}^\infty
              \frac{d\sigma\,\rho_{xy}(\sigma)}{\sigma^3(\sigma-z)}}.
                                                               \tag{12}
\]

At infinity the subtracted integrand is O(σ⁻²), and at threshold it is integrable. The subtraction does not change the discontinuity. With u=√(1−4m²/σ), dσ=8m²u du/(1−u²)², independently obtain

\[
 \boxed{\widehat\chi_R(z)=\frac{z^3}{960\pi^2}
            \int_0^1\frac{u^6\,du}{4m^2-z(1-u^2)}}.           \tag{13}
\]

Equations (12),(13), with the specified boundary value, are exact integral representations. Above threshold the real part is a principal value, not an ordinary integral through a pole. The first low-energy terms, derived by a convergent series for |z|<4m², are

\[
 \widehat\chi_R(z)=\frac{z^3}{26880\pi^2m^2}
                  +\frac{z^4}{483840\pi^2m^4}+O(z^5/m^6).
                                                               \tag{14}
\]

Each χ has mass dimension four; a_0 has dimension four, a_1 dimension two, a_2 dimension zero. Setting the three Taylor coefficients of **the remainder** to zero allocates these terms to local couplings; it does not predict their measured values. Higher terms in (14) are fixed scalar-loop contributions in this scheme, not a complete prediction for all possible independent higher-derivative gravitational operators.

## 5. Causality, positivity and physical absorption

For Im ω>0, (12) is analytic: σ+k²−ω² cannot vanish on the integration domain. Hermiticity gives χ_R(−ω,k)=χ_R(ω,k)*. The physical cut starts at |ω|=√(k²+4m²); using (9) or the pole u=β in (13) yields

\[
 \operatorname{Im}\chi_R(\omega,k)
 =\pi\,\operatorname{sgn}(\omega)\rho_{xy}(\omega^2-k^2).
                                                               \tag{15}
\]

For a long weak monochromatic metric drive h=H cosωt, the mean energy input into matter per volume is

\[
 \overline{\mathcal P}=-\overline{\dot h\,\delta\langle O\rangle}
       =\tfrac12\omega H^2\operatorname{Im}\chi_R(\omega,k)\ge0
       \quad(\omega>0).                                      \tag{16}
\]

The finite-switch response is a convolution with this spectrum, not an exactly monochromatic threshold experiment. The local real polynomial contributes reactive/contact response, not steady absorptive power. Nonzero absorption describes scalar pair production by an externally supplied gravitational-wave-like source; it does not require a dynamical metric. On-shell massless tensor waves have s=0 and no vacuum pair-production cut for m>0.

**Actual support proof.** Free-field microcausality implies [T(x),T(y)]=0 at spacelike separation, so (5) vanishes there and for x⁰<y⁰ before any Fourier simplification. Equivalently in the reduced (t,z) channel (12) is a subtracted positive superposition of massive Klein–Gordon retarded fundamental distributions, each supported in the future light cone. Covariant local extensions add derivatives of δ, not spacelike tails. These are distributional statements after removal of the regulator and smearing; they do not follow from a hard frequency filter. Thus causal matter response is established. Positivity of (1) establishes passive vacuum absorption in this channel, **not stability of an independently supplied metric equation**, complete quantum gravity or a fundamental time arrow. The retarded preparation/contour is an explicit operational boundary input; time reversal transforms it into the advanced response.

## 6. Independent checks, source comparison and limitations

`evidence/derive_response.py` derives the phase-space normalization, angular factor, identical-particle factor, Ward coefficient, subtraction and variable-change identities, oscillator sign, low-energy coefficients and metric seagull (15 exact checks). `evidence/check_independent.py` computes the Wick shell directly, integrates (12) and (13) separately, checks a boosted full vertex and approaches the branch cut numerically. At s=9m² the relative absorption error decreases from 0.00304 to 0.0000304 as the imaginary invariant decreases from 10⁻²m² to 10⁻⁴m². These are consistency illustrations, not empirical measurements or proofs by unit tests. The analytic arguments above establish support and domains.

Primary benchmark: Anderson, Molina-París & Mottola, [arXiv:gr-qc/0209075](https://arxiv.org/pdf/gr-qc/0209075), manuscript pp.7–11, Eqs.(3.4)–(3.10),(4.2)–(4.4); pp.19–21, Eqs.(B37a),(B39)–(B47). Their tensor density is ρ^(T)=s²β⁵/(960π²). Its xy,xy projector entry is 1/2; hence our independently computed (1) is ρ^(T)/2. Their Π sign and metric-variation convention must be translated using (3)–(5). This comparison validates normalization; no gravitational theorem is borrowed into our one-channel calculation.

No continuum wedge entropy, scalar K_eff, Hilbert projector, Schur projector or PT grading appears in this calculation. Neither I03's Euclidean local terms nor I01's thermal restrictions substitute for a real-time response. Reports 02–03 state exactly which dynamical premises remain independent.
