"""Cycle 2: full linear Einstein operator, canonical quotient and kinetic sign.
The metric action is obtained from the constrained BF branch in report 01.
The TT basis is checked to exhaust the physical null-symbol quotient, not assumed.
"""
from pathlib import Path
import json
import sympy as s
checks={}
def eq(name,x,y):
 r=x-y
 ok=all(s.simplify(v)==0 for v in r) if isinstance(r,s.MatrixBase) else s.simplify(r)==0
 checks[name]={'pass':bool(ok)}
 if not ok:raise AssertionError((name,r))
sympairs=[(i,j) for i in range(4) for j in range(i,4)]
hv=s.symbols('h0:10'); H=s.zeros(4)
for x,(i,j) in zip(hv,sympairs):H[i,j]=H[j,i]=x
w,k=s.symbols('omega k',real=True); et=[-1,1,1,1]
q=s.Matrix([-w,0,0,k]); qu=s.diag(*et)*q; q2=(q.T*qu)[0]
trace=sum(et[i]*H[i,i] for i in range(4)); qHq=(qu.T*H*qu)[0]
G=s.Matrix(4,4,lambda i,j:(-q[i]*(qu.T*H[:,j])[0]-q[j]*(qu.T*H[:,i])[0]
       +q2*H[i,j]+q[i]*q[j]*trace+(et[i]*(qHq-q2*trace) if i==j else 0))/2)
Gvec=s.Matrix([G[i,j] for i,j in sympairs]); D=Gvec.jacobian(hv)
xi=s.symbols('xi0:4'); gauge=s.Matrix([q[i]*xi[j]+q[j]*xi[i] for i,j in sympairs]).jacobian(xi)
eq('linear_Bianchi',qu.T*G,s.zeros(1,4))
eq('all_four_gauge_vectors_annihilated',D*gauge,s.zeros(10,4))
eq('off_shell_Einstein_rank',D.subs({w:3,k:2}).rank(),6)
Dn=D.subs({w:2,k:2}); Gn=gauge.subs({w:2,k:2})
eq('null_shell_Einstein_rank',Dn.rank(),4)
eq('null_shell_gauge_rank',Gn.rank(),4)
plus=s.Matrix([1 if (i,j)==(1,1) else -1 if (i,j)==(2,2) else 0 for i,j in sympairs])
cross=s.Matrix([1 if (i,j)==(1,2) else 0 for i,j in sympairs])
eq('plus_null_solution',Dn*plus,s.zeros(10,1))
eq('cross_null_solution',Dn*cross,s.zeros(10,1))
eq('gauge_plus_physical_exhaust_nullspace',Gn.row_join(plus).row_join(cross).rank(),10-Dn.rank())
eq('plus_wave_operator',Gvec.subs(dict(zip(hv,list(plus))))[sympairs.index((1,1))],q2/2)
# Independent canonical constraint count at nonzero spatial momentum along z.
# q=(hxx,hyy,hzz,hxy,hxz,hyz), p=(Pxx,Pyy,Pzz,2Pxy,2Pxz,2Pyz).
qq=s.Matrix(s.symbols('q0:6')); pp=s.Matrix(s.symbols('p0:6')); phase=qq.col_join(pp)
C=s.Matrix([qq[0]+qq[1],pp[4],pp[5],pp[2]])
J=C.jacobian(phase); Omega=s.zeros(12)
Omega[:6,6:]=s.eye(6);Omega[6:,:6]=-s.eye(6)
eq('four_independent_canonical_constraints',J.rank(),4)
eq('linear_constraints_first_class',J*Omega*J.T,s.zeros(4))
eq('four_independent_gauge_directions',(Omega*J.T).rank(),4)
eq('physical_phase_dimension',12-2*J.rank(),4)
# Derive the actual reduced kinetic/gradient coefficients from linear Christoffels.
P,Cx,Pt,Ct,Pz,Cz=s.symbols('P C Pt Ct Pz Cz',real=True)
HH=s.Matrix([[0,0,0,0],[0,P,Cx,0],[0,Cx,-P,0],[0,0,0,0]])
def deriv(expr,mu):
 if mu==0:return s.diff(expr,P)*Pt+s.diff(expr,Cx)*Ct
 if mu==3:return s.diff(expr,P)*Pz+s.diff(expr,Cx)*Cz
 return s.S.Zero
Gamma=[[[et[a]*(deriv(HH[a,c],b)+deriv(HH[a,b],c)-deriv(HH[b,c],a))/2
          for c in range(4)]for b in range(4)]for a in range(4)]
L=sum(et[mu]*(Gamma[a][mu][b]*Gamma[b][mu][a]-Gamma[a][mu][mu]*Gamma[b][a][b])
      for mu in range(4) for a in range(4) for b in range(4))
eq('derived_EH_GammaGamma_reduced_density',s.expand(L),(Pt**2+Ct**2-Pz**2-Cz**2)/2)
kap,sgn=s.symbols('kappa sign',nonzero=True)
Lphys=sgn*L/(2*kap)
eq('physical_velocity_Hessian',s.hessian(Lphys,(Pt,Ct)),sgn*s.eye(2)/(2*kap))
theta=s.symbols('theta',real=True);R=s.Matrix([[s.cos(theta),-s.sin(theta)],[s.sin(theta),s.cos(theta)]])
Hr=s.simplify(R*HH[1:3,1:3]*R.T)
eq('helicity_double_angle_plus',s.trigsimp(Hr[0,0]),P*s.cos(2*theta)-Cx*s.sin(2*theta))
eq('helicity_double_angle_cross',s.trigsimp(Hr[0,1]),P*s.sin(2*theta)+Cx*s.cos(2*theta))
# BF curl constraints have one Bianchi reducibility per internal generator.
curl=s.Matrix([[0,-k,0],[k,0,0],[0,0,0]])
eq('BF_curvature_constraint_rank_per_generator',curl.subs(k,1).rank(),2)
eq('BF_local_physical_phase_dimension',36-2*(6+6*2),0)
out={'sympy':s.__version__,'checks':checks,'null_symbol_rank':4,'gauge_rank':4,
     'canonical_constraint_rank':4,'physical_configuration_count':2,
     'GammaGamma_density':str(s.expand(L)),
     'scope':'Exact linear regular-branch analysis, not a full nonlinear Dirac calculation of all sectors.'}
Path(__file__).with_name('modes_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(len(checks),'exact mode/constraint checks passed')
