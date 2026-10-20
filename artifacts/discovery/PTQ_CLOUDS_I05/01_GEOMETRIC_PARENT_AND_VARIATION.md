# I05 — one real constrained-BF construction

**Constructed benchmark:** real non-chiral Lorentzian Plebanski gravity, compared with its unconstrained BF control. Simplicity is an explicit additional geometric postulate. It admits a Palatini gravitational branch; neither the postulate nor selection of that branch is derived from observer restriction or matter fluctuations. No Einstein–Hilbert term is inserted into the starting action.

## 1. Field, reality and boundary domain

Work on an oriented four-manifold M; for the local calculations choose a contractible patch of R⁴. The principal bundle has group SO⁺(1,3); its associated real vector bundle E has fixed internal η_IJ=diag(−1,1,1,1) and ε_0123=+1. The independent fields are

\[
 A^{IJ}=-A^{JI}\in\Omega^1(M,\operatorname{ad}P),\quad
 F^{IJ}=dA^{IJ}+A^I{}_K\wedge A^{KJ},\quad
 B^{IJ}=-B^{JI}\in\Omega^2(M,\operatorname{ad}P),             \tag{1}
\]

plus a real scalar bundle tensor φ_IJKL with antisymmetry in each pair and symmetry under pair exchange (21 components), and a real four-form ν. All fields are real. A^I{}_J=A^{IK}η_KJ is a metric-compatible Lorentz connection, not an unrestricted affine connection. No independent scalar/one-form compensator, affine projective symmetry, Immirzi deformation or self-dual complex reality condition is added.

The internal Hodge map is

\[
 (\star X)^{IJ}=\tfrac12\epsilon^{IJ}{}_{KL}X^{KL},\quad
 \epsilon^{0123}=-1,\quad \star^2=-I.                        \tag{2}
\]

It acts on internal bivector indices, not on spacetime form degree; a spacetime metric is not yet supplied. A real coframe and its spacetime Hodge star are reconstructed only on an appropriate nondegenerate constrained branch. Mistaking (2) for a real self-duality projector would be an error.

For the variation use smooth fields and compactly supported variations; equivalently on a finite region fix the pullback of δA=0 at its boundary. This removes the BF boundary variation (1/2κ)∫_∂M B^{IJ}∧δA_IJ. Gauge transformations used in the local reduction vanish at the boundary or have compact support. Boundary charges, edge modes and global holonomies are excluded from the **local bulk** count, not proved absent. A future-directed time orientation/retarded preparation is selected only for the Lorentzian background response in report 02. No quantum functional measure is constructed.

## 2. Complete actions and all independent variations

**CONTROL:**

\[
 S_{BF}[A,B]=\frac1{2\kappa}\int_M B^{IJ}\wedge F_{IJ}.       \tag{3}
\]

**CANDIDATE DEFINITION / independent simplicity assumption:**

\[
 \boxed{S_P[A,B,\phi,\nu]=\frac1{2\kappa}\int_M
 \left[B^{IJ}\wedge F_{IJ}
 -\frac12\phi_{IJKL}B^{IJ}\wedge B^{KL}
 -\frac\nu2\epsilon^{IJKL}\phi_{IJKL}\right].}              \tag{4}
\]

Repeated antisymmetric pairs are summed over **all** ordered indices. κ is an independent nonzero constant; choose κ>0 for comparison to positive-energy matter later. If coframe one-forms carry length, [B]=length², [κ]=length², [φ]=length⁻² and [ν]=length⁴. The ν term implements the epsilon-trace condition instead of silently restricting multiplier variations. The vacuum cosmological parameter is set to zero in this model, so a flat branch can be studied. This is a model input, not an explanation of physical Λ. No field-dependent constitutive function or tuned coefficient-locking rule is present.

Integration by parts and independent variations give

\[
 \begin{array}{rcl}
 \delta A &: & D_A B^{IJ}=0,\\
 \delta B &: & F_{IJ}=\phi_{IJKL}B^{KL},\\
 \delta\phi &: & B^{IJ}\wedge B^{KL}+\nu\epsilon^{IJKL}=0,\\
 \delta\nu &: & \epsilon^{IJKL}\phi_{IJKL}=0.
 \end{array}                                                  \tag{5}
\]

The first equation comes from δF=D_AδA. Pair-exchange symmetry makes the two B variations of the quadratic term equal. The δφ equation has the same pair symmetries, so all 21 independent multiplier equations are included. Contracting it with ε_IJKL, using ε_IJKLε^IJKL=−24, gives

\[
 \nu=V_B:=\frac1{24}\epsilon_{IJKL}B^{IJ}\wedge B^{KL}.       \tag{6}
\]

Thus eliminating ν leaves 20 algebraic simplicity conditions on the 36 components of B. The constraint fixes the **form** of B, not a unique coframe or metric. In the control, varying the same independent A,B gives F=0 and D_AB=0, with no simplicity restriction.

## 3. Nondegenerate algebraic reconstruction and its limitations

At a point, let W denote the symmetric wedge pairing on Λ²T* (signature (3,3) relative to a chosen four-volume). In the ordered bases (01,02,03,12,13,23), the matrix b of B satisfies

\[
 bWb^T=-\nu\,\epsilon^{[IJ][KL]}.                            \tag{7}
\]

For ν≠0 this is invertible. It sends the null cone of one wedge pairing to the other. In dimension four a two-form u is decomposable iff u∧u=0: in matrix language its Pfaffian vanishes and its rank is ≤2. Hence b maps the cone of simple two-forms to itself. Its maximal three-dimensional null planes have two forms: ℓ∧V and Λ²H for a line ℓ or three-plane H. Their intersection relations reconstruct the underlying projective four-space. An invertible real linear map preserving the cone either preserves these two families or interchanges them. In the former case it is a scalar multiple of an exterior-square map Λ²e; in the latter composing with the internal star converts it to the former case. The positive scalar magnitude can be absorbed into a real coframe, leaving an overall ± sign.

This gives the local, real, nondegenerate solutions

\[
 \mathrm I_\pm:\ B^{IJ}=\pm e^I\wedge e^J,
 \qquad
 \mathrm {II}_\pm:\ B^{IJ}=\pm\star(e\wedge e)^{IJ},
 \qquad \det e\ne0.                                         \tag{8}
\]

This is an algebraic classification, not four established universes or four physically equivalent states. Global existence of a single coframe requires the corresponding bundle/topology; the proof and propagation calculation are local. Discrete orientation/time-orientation choices are not automatically in SO⁺ gauge. Formula (8) does not assert that the four labels exhaust global connected components after all such choices.

For vol_e=e⁰∧e¹∧e²∧e³,

\[
 V_{e\wedge e}=\operatorname{vol}_e,\qquad
 V_{\star(e\wedge e)}=-\operatorname{vol}_e.                  \tag{9}
\]

The signs in (9) follow from the **Lorentzian** identity star²=−1 and are explicitly checked, rather than imported from a Euclidean formula. All reconstructed metrics g=η_IJ e^I⊗e^J have Lorentzian signature; all original fields and e can be real without chiral reality constraints.

For an independent local check, linearize the 21 components of C(B)=B∧B+V_Bε at B̄=star(dx∧dx). Their Jacobian has exact rank 20. The coframe map δe↦star(δe∧ē+ē∧δe) has rank 16 and lies in its kernel, whose dimension is 36−20=16. The two spaces are therefore equal. These ranks are recomputed with both SymPy and an independent rational-elimination implementation, including a non-diagonal real coframe. Thus no unconstrained B fluctuation survives unnoticed on this regular branch. This rank is **not** the gravitational physical degree count; gauge and dynamical constraints are still required.

## 4. Deriving the Palatini branch and connection equation

On any regular branch the multiplier terms vanish on the constraints: φ_IJKL B^{IJ}∧B^{KL}=−ν ε^{IJKL}φ_IJKL=0. Since the constraint rank is regular, solving it before varying along its coframe tangent is legitimate holonomic reduction. The remaining normal equations determine the multipliers; they are not lost metric equations.

On branch II_σ, σ=±1,

\[
 S_{II_\sigma}[e,A]=\frac{\sigma}{4\kappa}
                  \int\epsilon_{IJKL}e^I\wedge e^J\wedge F^{KL}.
                                                               \tag{10}
\]

This is a **derived Palatini-like action**, not the starting definition. Varying its independent fields yields

\[
 \epsilon_{IJKL}T^K\wedge e^L=0,\qquad
 \epsilon_{IJKL}e^J\wedge F^{KL}=0,\quad
 T^I=de^I+A^I{}_J\wedge e^J.                                \tag{11}
\]

For invertible e, the first equation enforces T^I=0, not a prescribed trace-only torsion ansatz. One direct verification is that the 24×24 map T↦D(e∧e) and the 24×24 map from a metric-compatible contorsion to T both have rank 24 in an orthonormal coframe. Frame covariance transfers this invertibility to any invertible coframe; internal star is also invertible. Thus A is uniquely the torsion-free spin connection up to its ordinary frame transformation. There is no affine-projective null direction or independent connection wave in this branch with spinless sources.

Write F^{IJ}=(1/2)R^{IJ}{}_{KL} e^K∧e^L. Contraction gives ε_IJKL e^I∧e^J∧F^{KL}=2R vol_e. Consequently, after solving the auxiliary connection,

\[
 S_{II_\sigma}[e,A(e)]=\frac{\sigma}{2\kappa}
                            \int\operatorname{vol}_e\,R[g]. \tag{12}
\]

The coframe equation in (11) is G_μν=0. This Einstein form has been obtained from (4) plus the simplicity postulate and the regular branch, not postulated as the candidate's first action. It does not explain why nature imposes (4). For the perturbative physical calculation take vol_e positive in the fixed orientation and a future-directed e⁰; then σ=+ gives the usual positive EH coefficient with κ>0. If coframe orientation is reversed, the sign of the density relative to the fixed integral reverses as well. Health concerns the **combined** action/orientation sign relative to fixed positive-energy matter, not an unqualified branch label.

## 5. Other sectors are retained

**Regular I_σ branch (σ=±1):** its reduced action is σ(1/2κ)∫e^I∧e^J∧F_IJ. The connection equation again gives T=0. The exact Cartan identity

\[
 d(e^I\wedge T_I)=T^I\wedge T_I-e^I\wedge e^J\wedge F_{IJ}  \tag{13}
\]

then makes the action vanish after eliminating the connection, modulo actual boundaries. The coframe equation e^J∧F_IJ=0 is the torsion-free first Bianchi identity, not Einstein's equation. Indeed every smooth invertible coframe with its torsion-free connection solves this sector: take φ_IJKL=(σ/2)R_IJKL, whose pair symmetry and epsilon trace follow from Riemann symmetry/Bianchi. Thus this branch has no ordinary bulk metric kinetic term or graviton propagator; arbitrary coframe functions are not propagating Einstein waves. We do not simply discard it. Equation (13) is ordinary Nieh–Yan, not a Hodge torsion norm or a historical CS+ equivalence.

**Degenerate branch ν=0:** B^{IJ}∧B^{KL}=0 for all pairs. The span of the six spacetime two-forms is isotropic for W and has dimension at most three. Such a span is contained in a maximal null plane of the line/three-plane types used above. It need not define an invertible coframe, metric, causal cone or time direction. An exact nonzero example is A=0,φ=0,ν=0,B^{01}=dx⁰∧dx¹, all other independent B components zero. B=0,A=0,φ=0 is another solution. These solve **all** (5), not only simplicity.

At B=0 the eliminated simplicity Jacobian has rank zero; at the rank-one example it has rank six, versus 20 on the regular branch. Linearizing the zero background misses quadratic conditions b∧b=0 and gives BF-like equations da=0,db=0. Such linearization is singular and does not prove a common nonlinear degree count for every degenerate stratum. No GR coframe inversion, healthy propagator, or two-mode statement is extended to this sector. A complete nonlinear canonical stratification of degenerate sectors is not executed or claimed in this bounded sprint.

## 6. Provenance and evidence

The displayed variation and exterior algebra were executed in `evidence/derive_simplicity.py` (21 exact checks); `check_independent.py` uses separate dictionary-form arithmetic and rational elimination. The pointwise rank computations verify local regularity, not global topology or a quantum state count.

Historical calibration, not novelty: De Pietri & Freidel, [gr-qc/9804071v2](https://arxiv.org/pdf/gr-qc/9804071), §III, Eqs.(5)–(8) and Theorem 1, Eqs.(17)–(23), PDF pp.4–6, establishes the Euclidean four-sector warning; we explicitly compute the real Lorentzian star/signs instead of transferring its signature. Alexandrov & Krasnov, [0809.4763](https://arxiv.org/pdf/0809.4763), §2, treats full Lorentz-group constraints. Their nonlinear canonical result is a comparison source, not a calculation rerun here. Reports 02–03 give the independently executed linear physical-mode result and the remaining first-principles gap.
