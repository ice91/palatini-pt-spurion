"""NEW D02 cycle 2: two motivated maps of ONE fixed action/operator and source protocol."""
from pathlib import Path
import json
import sympy as s
HERE=Path(__file__).resolve().parent
k=s.symbols('k',positive=True);a,b=s.symbols('a b',positive=True);z=s.symbols('z',real=True)
kap=k*k;c=(1+kap)/2;h=(kap-1)/2
u=s.Matrix([1,1])/s.sqrt(2);w=s.Matrix([1,-1])/s.sqrt(2)
K=s.eye(2)+(kap-1)*u*u.T;M=s.diag(a,b);D=z*K-M
n1=s.Matrix([1,0])/s.sqrt(c)
q1=s.Matrix([-h/(s.sqrt(c)*k),s.sqrt(c)/k])
n2=u/k;q2=w
T1=s.Matrix.hstack(n1,q1);T2=s.Matrix.hstack(n2,q2)
checks={};expr={}
def zero(name,x):
 vals=list(x) if isinstance(x,s.MatrixBase) else [x]
 checks[name]=all(s.simplify(v)==0 for v in vals)
 assert checks[name],(name,x)
def put(name,x):expr[name]=str([s.simplify(v) for v in x] if isinstance(x,(list,tuple)) else s.simplify(x))
put('K',K);put('D_same_parent',D);put('detD',s.factor(D.det()))
put('poles',s.solve(D.det(),z))
put('PT_all_real',s.conjugate(K)-K)
# Source and detector are fixed ONCE, before either chart is chosen.
J=K*n1
Rfull=s.factor((J.T*D.inv()*J)[0])
put('fixed_source_J',J);put('fixed_detector_Jtranspose',J.T);put('matched_response_direct',Rfull)
zero('fixed_source_norm', (J.T*K.inv()*J)[0]-1)
Ps=[];Ds=[]
for i,(T,n) in enumerate([(T1,n1),(T2,n2)],1):
 zero('chart%d_canonical_kinetic'%i,T.T*K*T-s.eye(2))
 P=s.simplify(n*n.T*K);Ps.append(P)
 zero('projector%d_idempotent'%i,P*P-P)
 zero('projector%d_K_selfadjoint'%i,P.T*K-K*P)
 zero('projector%d_real_PT_compatible'%i,s.conjugate(P)-P)
 H=s.simplify(T.T*M*T);Di=s.simplify(T.T*D*T);Ds.append(Di)
 A,B,F=Di[0,0],Di[0,1],Di[1,1]
 E=s.factor(A-B*B/F)
 src=s.simplify(T.T*J);sr=s.simplify(src[0]-B/F*src[1])
 response=s.factor(sr*sr/E+src[1]**2/F)
 zero('matched_response_chart%d_equals_direct'%i,response-Rfull)
 zero('all_source_detector_matrix_identity_chart%d'%i,T*Di.inv()*T.T-D.inv())
 zero('Schur_determinant_identity_chart%d'%i,E*F-D.det()/kap)
 put('P%d'%i,P);put('T%d'%i,T);put('mass_H%d'%i,H)
 put('operator_D%d'%i,Di);put('strict_A%d'%i,A);put('exact_Schur_E%d'%i,E)
 put('transformed_source%d'%i,src);put('reduced_source%d'%i,sr)
 put('source_contact%d'%i,s.factor(src[1]**2/F))
 put('strict_matched_source_response%d'%i,s.factor(src[0]**2/A))
 put('closure_commutator%d'%i,K.inv()*M*P-P*K.inv()*M)
 if i==1:
  put('F1_zero_full_determinant',s.factor(D.det().subs(z,H[1,1])))
 else:
  put('F2_zero_full_determinant',s.factor(D.det().subs(z,H[1,1])))
  intrinsic=s.factor((n.T*K*D.inv()*K*n)[0])
  put('intrinsic_bright_source_response',intrinsic)
  put('intrinsic_bright_minus_fixed_species',s.factor(intrinsic-Rfull))
put('projectors_distinct',Ps[0]-Ps[1])
# The transition is orthogonal for the same physical kinetic pairing.
O=s.simplify(T2.inv()*T1)
zero('chart_transition_orthogonal',O.T*O-s.eye(2))
zero('same_parent_operator_transition',Ds[1]-O*Ds[0]*O.T)
put('chart_transition_O',O)
# Actual singular kinetic locus, without dividing by k.
BB,DD,Jb,Jd=s.symbols('B D Jb Jd')
ss=(a+b)/2;dd=(a-b)/2
Bsol=s.solve(-ss*BB-dd*DD-Jb,BB)[0]
Dreduced=s.factor((z-ss)*DD-dd*Bsol-Jd)
put('kappa_zero_bright_constraint',Bsol)
put('kappa_zero_dark_equation',Dreduced)
put('kappa_zero_dark_mass',s.factor(ss-dd**2/ss))
# Independent analytic discriminant expression with manifest positive pieces.
disc=s.expand(c*c*(a+b)**2-4*kap*a*b)
zero('discriminant_positive_decomposition',disc-c*c*(a-b)**2-a*b*(kap-1)**2)
# General matched-source identity, not specific to the chosen source.
x,y=s.symbols('Jx Jy',real=True);Js=s.Matrix([x,y]);T=T2
Di=s.simplify(T.T*D*T);v=T.T*Js
E=Di[0,0]-Di[0,1]**2/Di[1,1]
allsource=(v[0]-Di[0,1]*v[1]/Di[1,1])**2/E+v[1]**2/Di[1,1]
zero('all_sources_generate_same_response',allsource-(Js.T*D.inv()*Js)[0])
put('torsion_detector_source_form',s.Matrix([1,1]).T*D.inv()*J/s.sqrt(2))
res={'sympy':s.__version__,'parent_fixed':'K and diag(a,b), background and J identical for both maps; k=sqrt(kappa)>0','checks':checks,'all_pass':all(checks.values()),'expressions':expr}
(HERE/'projection_results.json').write_text(json.dumps(res,indent=2))
print(json.dumps({'checks':len(checks),'all_pass':res['all_pass'],'expressions':expr},indent=2))
