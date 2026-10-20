"""Exact first-cycle closure test. No spectrum is inferred from this calculation."""
from pathlib import Path
import json
from itertools import combinations
import sympy as s
OUT=Path(__file__).resolve().parent
pairs=list(combinations(range(4),2))
W=s.Matrix(6,6,lambda i,j:s.LeviCivita(*pairs[i],*pairs[j]))
D=s.diag(-1,-1,-1,1,1,1); S=D*W
checks={}
def check(name,condition):
    checks[name]=bool(condition)
    if not checks[name]: raise AssertionError(name)
def zero(M): return all(s.simplify(x)==0 for x in M)
def ext(e):
    return s.Matrix(6,6,lambda a,b:e[pairs[a][0],pairs[b][0]]*e[pairs[a][1],pairs[b][1]]-e[pairs[a][0],pairs[b][1]]*e[pairs[a][1],pairs[b][0]])
def proj(G):
    v=s.trace(W*G)/6; Q=2*s.trace(D*G)
    return s.simplify(G-v*W-Q*D/12)
def data(b):
    G=s.simplify(b*W*b.T); v=s.simplify(s.trace(W*G)/6); Q=s.simplify(2*s.trace(D*G))
    return G,v,Q,proj(G)
check('star_squared_minus_identity',S*S==-s.eye(6))
check('wedge_signature_3_3',list(W.eigenvals().values())==[3,3])
e1=s.eye(4); e2=s.diag(2,3,1,1)
b1=S*ext(e1); b2=S*ext(e2)
for n,b,det in [(1,b1,1),(2,b2,6)]:
    G,v,Q,C=data(b)
    check(f'micro_{n}_hard_simplicity',zero(G-v*W) and Q==0 and v==-det)
w=s.symbols('w',real=True)
b=s.simplify((1-w)*b1+w*b2); G,v,Q,C=data(b)
t1=1+5*w; t2=(1+w)*(1+2*w); delta=s.factor(t1-t2)
check('complement_difference',s.simplify(delta-2*w*(1-w))==0)
check('mean_Q_zero',Q==0)
check('mean_v',s.factor(v+(t1+2*t2)/3)==0)
check('mean_det_regular',s.factor(b.det())==(w+1)**2*(2*w+1)**2*(5*w+1))
check('nineteen_residual_nonzero_polynomial',not zero(C))
check('C19_norm_formula',s.factor(s.trace((C*W)**2)-s.Rational(4,3)*delta**2)==0)
Db=b2-b1
check('exact_covariance_identity',zero(G-((1-w)*data(b1)[0]+w*data(b2)[0]-w*(1-w)*Db*W*Db.T)))
check('normal_residual_covariance',zero(C+w*(1-w)*proj(Db*W*Db.T)))
half={w:s.Rational(1,2)}
bh=b.subs(half); Gh,vh,Qh,Ch=data(bh)
I=s.simplify(s.trace((Ch*W)**2)/vh**2)
check('half_v_minus_19_over_6',vh==-s.Rational(19,6))
check('half_det_63_over_2',bh.det()==s.Rational(63,2))
check('half_invariant_12_over_361',I==s.Rational(12,361))
check('beta_zero_is_not_reconstruction',Qh==0 and not zero(Ch))
beta=s.symbols('beta',real=True)
check('I06_scalar_benchmark',data(S+beta*s.eye(6))[2]==24*beta)
# Same map and action, a second microscopic coframe witness. These are not models.
b3=S*ext(s.diag(2,1,3,1)); C2=data((b1+b3)/2)[3]
normal_rank=s.Matrix.hstack(s.Matrix(Ch).reshape(36,1),s.Matrix(C2).reshape(36,1)).rank()
check('at_least_two_independent_C19_defects',normal_rank==2)
u,z=s.symbols('u z',real=True)
near=(1-w)*b1+w*S*ext(s.diag(1+u,1+z,1,1))
nearG,nearv,nearQ,nearC=data(near)
check('no_linear_amplitude_defect_u',zero(nearC.diff(u).subs({u:0,z:0})))
check('no_linear_amplitude_defect_z',zero(nearC.diff(z).subs({u:0,z:0})))
check('second_order_mixed_defect',not zero(nearC.diff(u,z).subs({u:0,z:0})))
# Nonconstant proper Lorentz frames can be checked fiberwise. A constant rational
# witness below verifies different cell and root transformations, including links.
L=s.Matrix([[s.Rational(5,4),s.Rational(3,4),0,0],[s.Rational(3,4),s.Rational(5,4),0,0],[0,0,1,0],[0,0,0,1]])
Rot=s.Matrix([[1,0,0,0],[0,s.Rational(3,5),-s.Rational(4,5),0],[0,s.Rational(4,5),s.Rational(3,5),0],[0,0,0,1]])
eta=s.diag(-1,1,1,1)
check('proper_orthochronous_frames',L.T*eta*L==eta and Rot.T*eta*Rot==eta and L.det()==1 and Rot.det()==1 and L[0,0]>0)
R0=ext(L*Rot); R1=ext(L); R2=ext(Rot)
check('lorentz_wedge_invariance',R0*W*R0.T==W)
check('lorentz_star_equivariance',R0*S==S*R0)
mean_transformed=(1-w)*(R0*R1.inv())*(R1*b1)+w*(R0*R2.inv())*(R2*b2)
check('independent_frame_transport_covariance',zero(mean_transformed-R0*b))
check('C19_tensor_covariance',zero(data(R0*b)[3]-R0*C*R0.T))
check('defect_invariant_under_root_frame',s.simplify(s.trace((data(R0*bh)[3]*W)**2)/vh**2-I)==0)
# A smooth connected exact hard solution, not independent incompatible cells:
# e0=d[f(x)t], e1=g(x)dx, e2=dy,e3=dz, A=phi=0, nu=-f*g vol.
t,x,y,zcoord=s.symbols('t x y z',real=True); coords=[t,x,y,zcoord]
f=s.Function('f')(x); g=s.Function('g')(x)
e=s.Matrix([[f,t*s.diff(f,x),0,0],[0,g,0,0],[0,0,1,0],[0,0,0,1]]); bg=S*ext(e)
T=[]
for I0 in range(4):
 for mu,nu in pairs:
  T.append(s.diff(e[I0,nu],coords[mu])-s.diff(e[I0,mu],coords[nu]))
check('connected_parent_torsion_zero',all(a==0 for a in T))
dB=[]
for I0 in range(6):
 for a,c,d in combinations(range(4),3):
  def coeff(i,j):
   return bg[I0,pairs.index((i,j))] if i<j else -bg[I0,pairs.index((j,i))]
  dB.append(s.diff(coeff(c,d),coords[a])-s.diff(coeff(a,d),coords[c])+s.diff(coeff(a,c),coords[d]))
check('connected_parent_DB_zero',all(s.simplify(a)==0 for a in dB))
Gf,vf,Qf,Cf=data(bg)
check('connected_parent_algebraic_equations',zero(Cf) and Qf==0 and vf==-f*g)
# Remaining equations are identically F=phi B=0 and epsilon.phi=0 for A=phi=0.
results={'checks':checks,'number_passed':len(checks),'pair_order':pairs,'W':W.tolist(),'D':D.tolist(),'S':S.tolist(),
 'weighted_v':str(s.factor(v)),'weighted_det':str(s.factor(b.det())), 'complement_difference':str(delta),
 'weighted_C19':[[str(s.factor(a)) for a in row] for row in C.tolist()],
 'half_mean_b':[[str(a) for a in row] for row in bh.tolist()], 'half_C19':[[str(a) for a in row] for row in Ch.tolist()],
 'half_v':str(vh),'half_det':str(bh.det()),'half_Q':str(Qh),'half_invariant':str(I),
 'half_Gram_times_W_eigenvalues':{str(k):int(val) for k,val in (Gh*W).eigenvals().items()},'independent_normal_directions_lower_bound':normal_rank,
 'nearby_mixed_C19':[[str(s.factor(a)) for a in row] for row in nearC.diff(u,z).subs({u:0,z:0}).tolist()],
 'stop':'C19_CLOSURE_FAILURE: no beta spectrum or microscopic lambda calculation authorized downstream'}
(OUT/'closure_results.json').write_text(json.dumps(results,indent=2,default=str)+'\n')
print(json.dumps({'checks_passed':len(checks),'half_v':str(vh),'half_det':str(bh.det()),'half_invariant':str(I),'normal_rank_lower_bound':normal_rank,'stop':results['stop']},indent=2))
