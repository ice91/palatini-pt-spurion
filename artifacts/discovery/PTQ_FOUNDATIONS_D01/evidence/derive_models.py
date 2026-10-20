"""NEW CANDIDATE D01: action-derived two-scalar PT model; no original modules.
Run with installed palpt Python -B; all outputs local to this evidence directory.
"""
from pathlib import Path
import json
import sympy as s

HERE=Path(__file__).resolve().parent
results={}; checks={}
def put(k,v): results[k]=str(v)
def zero(k,x):
    vals=list(x) if isinstance(x,s.MatrixBase) else [x]
    checks[k]=all(s.simplify(v)==0 for v in vals)
    if not checks[k]: raise AssertionError((k,x))
a,b,g,z=s.symbols('a b g z', real=True)
I=s.eye(2); Th=s.diag(1,-1); sig2=s.Matrix([[0,-s.I],[s.I,0]])
M=s.Matrix([[a,s.I*g],[s.I*g,b]])
D=z*I-M
put('mass_matrix',M); put('Fourier_operator',D)
zero('anti_linear_PT_mass',Th*M.conjugate()*Th-M)
put('grading_commutator',Th*M-M*Th)
put('det_D_from_matrix',s.factor(D.det()))
put('mass_roots',s.solve(D.det(),z))
# Vary independent dual action, not an assigned equation.
t,x=s.symbols('t x',real=True)
p=s.Matrix([s.Function(n)(t,x) for n in ('p','q')])
u=s.Matrix([s.Function(n)(t,x) for n in ('up','uq')])
L=(u.diff(t).T*p.diff(t)-u.diff(x).T*p.diff(x)-u.T*M*p)[0]
EL=s.Matrix([s.diff(L,f)-s.diff(s.diff(L,s.diff(f,t)),t)-s.diff(s.diff(L,s.diff(f,x)),x) for f in u])
zero('variation_independent_dual',EL-(-p.diff(t,2)+p.diff(x,2)-M*p))
put('EL_from_independent_dual_action',EL)
# General PT restriction for a complex symmetric constant matrix.
ar,ai,br,bi,cr,ci=s.symbols('ar ai br bi cr ci',real=True)
Mc=s.Matrix([[ar+s.I*ai,cr+s.I*ci],[cr+s.I*ci,br+s.I*bi]])
put('general_PT_residual',s.simplify(Th*Mc.conjugate()*Th-Mc))
# Derive metric witness from the quasi-Hermiticity equation.
r=s.symbols('r',real=True)
Gr=I+r*sig2
metric_res=M.conjugate().T*Gr-Gr*M
rsol=s.solve(metric_res[0,1],r)[0]
G=s.simplify(Gr.subs(r,rsol))
put('metric_equation',metric_res);put('metric_solution_r',rsol);put('metric_witness_G',G)
zero('G_metric_compatibility',M.conjugate().T*G-G*M)
zero('PT_metric_covariance',Th*G.conjugate()*Th-G)
put('G_eigenvalues',G.eigenvals()); put('det_G',s.factor(G.det()))
# Controlled revision: real physical action obtained using G, same D equations.
v=s.Matrix(s.symbols('u v w y',real=True))
vt=s.Matrix(s.symbols('ut vt wt yt',real=True)); vx=s.Matrix(s.symbols('ux vx wx yx',real=True))
def complexify(v):return s.Matrix([v[0]+s.I*v[1],v[2]+s.I*v[3]])/s.sqrt(2)
psi,psit,psix=map(complexify,(v,vt,vx))
Lg=s.expand((psit.conjugate().T*G*psit-psix.conjugate().T*G*psix-psi.conjugate().T*G*M*psi)[0])
zero('physical_action_real',s.conjugate(Lg)-Lg)
K=s.simplify(s.hessian(Lg,vt)); V=s.simplify(-s.hessian(Lg,v))
put('physical_real_velocity_Hessian',K);put('physical_real_mass_Hessian',V)
put('kinetic_determinant',s.factor(K.det()))
MR=s.Matrix([[a,0,0,-g],[0,a,g,0],[0,-g,b,0],[g,0,0,b]])
zero('action_hessian_reproduces_mass',V-K*MR)
zero('spatial_Hessian_is_minus_kinetic',s.hessian(Lg,vx)+K)
zero('no_algebraic_constraints_regular_kinetic',K.det()-(1-4*g*g/(b-a)**2)**2)
put('real_principal_determinant',s.factor((z*K).det()))
# Eliminate the actual scalar complementary equation.
chi,J1,J2=s.symbols('chi J1 J2')
comp=s.solve(D[1,0]*chi+D[1,1]*s.Symbol('qfield')-J2,s.Symbol('qfield'))[0]
Deff=s.simplify(D[0,0]-D[0,1]*(1/D[1,1])*D[1,0])
put('eliminated_complement',comp); put('Schur_from_blocks',Deff)
zero('Schur_times_F_equals_full_determinant',s.factor(Deff*(z-b)-D.det()))
put('strict_inverse_11',1/(z-a));put('coordinate_full_inverse_11',s.factor(D.inv()[0,0]))
put('low_z_Schur',s.series(Deff,z,0,3))
# Physical source e1 and its G-dual detector; not bare coordinate projection.
e=s.Matrix([1,0]); physical=s.factor((e.conjugate().T*G*D.inv()*e)[0])
put('physical_source_detector_response',physical)
zero('physical_response_high_frequency_norm',s.limit(z*physical,z,s.oo)-1)
# After the controlled pairing revision the actual variational Hessian is H=G D.
H=s.simplify(G*D)
Heff=s.factor(H[0,0]-H[0,1]/H[1,1]*H[1,0])
put('physical_action_complex_Hessian_H',H)
put('physical_variational_Schur_Heff',Heff)
zero('physical_variational_Schur_det_identity',Heff*H[1,1]-G.det()*D.det())
source=G*e
j_eff=s.simplify(source[0]-H[0,1]/H[1,1]*source[1])
j_eff_left=s.simplify(s.conjugate(source[0])-s.conjugate(source[1])/H[1,1]*H[1,0])
contact=s.simplify(s.conjugate(source[1])*source[1]/H[1,1])
put('variational_reduced_source',j_eff)
put('variational_contact_response',contact)
zero('physical_response_from_variational_elimination',j_eff_left*j_eff/Heff+contact-physical)
put('Schur_F_coordinate_singularity_full_response',s.factor(physical.subs(z,b)))
# Adversarial analysis of bare-coordinate pole residues.
Delta,ss,c=s.symbols('Delta ss c',positive=True)
subs={a:c-Delta/2,b:c+Delta/2,g:g}
physical_gap=s.sqrt(Delta**2-4*g*g)
lp=c+physical_gap/2; lm=c-physical_gap/2
num_naive=z-(c+Delta/2)
put('bare_heavy_pole_residue',s.simplify(num_naive.subs(z,lp)/physical_gap))
num_phys=z-(c+Delta/2)+2*g*g/Delta
put('physical_heavy_pole_residue',s.simplify(num_phys.subs(z,lp)/physical_gap))
# Simplify residue using ss^2=Delta^2-4g^2, not an assumed output formula.
wp=s.simplify(((ss-Delta)/2+2*g*g/Delta)/ss)
wp=s.simplify(wp.subs(g*g,(Delta**2-ss**2)/4))
zero('positive_physical_residue_formula',wp-(Delta-ss)/(2*Delta))
put('physical_weights_gap_parameter',(s.simplify(1-wp),wp))
# Exact equivalent parameter pair g -> -g, including metric and source.
zero('sign_g_field_map_mass',M.subs(g,-g)-Th*M*Th)
zero('sign_g_field_map_metric',G.subs(g,-g)-Th*G*Th)
zero('sign_g_same_e1_source',Th*e-e)
zero('sign_g_same_physical_response',physical.subs(g,-g)-physical)
kappa=s.symbols('kappa',real=True)
put('inequivalent_pair_determinant_difference',s.factor(M.subs(g,2*kappa).det()-M.subs(g,kappa).det()))
# Explicit entire isospectral microscopic family, not claimed distinct physics.
eta=s.symbols('eta',real=True)
Mf=M.subs({a:c-ss*s.cosh(eta),b:c+ss*s.cosh(eta),g:ss*s.sinh(eta)})
Sf=s.cosh(eta/2)*I+s.sinh(eta/2)*sig2
Sf_inv=s.cosh(eta/2)*I-s.sinh(eta/2)*sig2
zero('isospectral_family_similarity',(Sf*Mf*Sf_inv-s.diag(c-ss,c+ss)).applyfunc(s.trigsimp))
zero('isospectral_metric_from_similarity',((Sf.conjugate().T*Sf)/s.cosh(eta)-(I+s.tanh(eta)*sig2)).applyfunc(s.trigsimp))
put('isospectral_parameter_family',Mf)
# A constant rephase removes the internal anti-linear grading (complete package transformed).
Rphase=s.diag(1,s.I)
zero('constant_rephase_removes_internal_PT_grading',Rphase*Th*Rphase.conjugate().inv()-I)
Mreal=s.simplify(Rphase*M*Rphase.inv()); Greal=s.simplify(Rphase*G*Rphase.inv())
zero('constant_rephase_real_mass',Mreal.conjugate()-Mreal)
zero('constant_rephase_real_pairing',Greal.conjugate()-Greal)
put('rephased_real_mass',Mreal);put('rephased_real_metric',Greal)
# Exceptional-point falsification by nilpotent Jordan witness.
Me=M.subs({a:c-ss,b:c+ss,g:ss}); N=Me-c*I
zero('exceptional_point_nilpotent',N*N)
checks['exceptional_point_not_diagonalizable']=N!=s.zeros(2) and N.rank()==1
put('exceptional_point_N',N)
# Above threshold, exact complex eigenvalues (parametric, not fitted data).
put('broken_PT_eigenvalues',M.subs({a:c-ss,b:c+ss,g:2*ss}).eigenvals())
# Weak-coupling coordinate heavy residue is negative, while physical ones positive.
assert all(checks.values())
(HERE/'derivation_results.json').write_text(json.dumps({'sympy_version':s.__version__,'expressions':results,'checks':checks,'all_pass':all(checks.values())},indent=2))
print(json.dumps({'checks':len(checks),'all_pass':all(checks.values()),'expressions':results},indent=2))
