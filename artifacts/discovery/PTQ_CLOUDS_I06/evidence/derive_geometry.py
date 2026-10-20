"""Cycle 1: rank-one multiplier softening and exact Lorentzian connection.
New code; prior sprint scripts/results are read-only and are not imported.
"""
from pathlib import Path
from itertools import combinations
import sympy as s
import json
P=list(combinations(range(4),2)); Q3=list(combinations(range(4),3))
eta=[-1,1,1,1]; W=s.Matrix(6,6,lambda a,b:s.LeviCivita(*P[a],*P[b]))
D=s.diag(*[eta[i]*eta[j]for i,j in P]); star=D*W
beta,lam=s.symbols('beta lambda',real=True,nonzero=True)
checks={}
def eq(name,a,b):
 r=a-b;ok=all(s.simplify(x)==0 for x in r)if isinstance(r,s.MatrixBase)else s.simplify(r)==0
 checks[name]={'pass':bool(ok)}
 if not ok:raise AssertionError((name,r))
def invs(B):
 G=B*W*B.T
 v=sum(W[i,j]*G[i,j]for i in range(6)for j in range(6))/6
 q=2*sum(D[i,i]*G[i,i]for i in range(6))
 return G,s.expand(v),s.expand(q)
def C19(B):
 G,v,q=invs(B);R=G-v*W-q*D/12
 return s.Matrix([s.expand(R[i,j])for i in range(6)for j in range(i,6)])
def ex2(e):
 return s.Matrix(6,6,lambda a,b:e[P[a][0],P[b][0]]*e[P[a][1],P[b][1]]-
                 e[P[a][0],P[b][1]]*e[P[a][1],P[b][0]])
B0=star+beta*s.eye(6)
eq('star_squared',star*star,-s.eye(6))
G,v,q=invs(B0)
eq('relaxed_Gram',G,(beta**2-1)*W+2*beta*D)
eq('volume',(v),beta**2-1);eq('q_contraction',q,24*beta)
eq('nineteen_constraints',C19(B0),s.zeros(21,1))
eq('B_invertibility',s.factor(B0.det()),(1+beta**2)**3)
tau=2*beta/(lam*(1-beta**2))
eq('soft_trace_equation',q+12*lam*v*tau,0)
eq('eliminated_multiplier_density',s.simplify(lam*v*tau**2/2),-2*beta**2/(lam*(1-beta**2)))
eq('trace_inversion_derivative',s.diff(tau,beta),2*(1+beta**2)/(lam*(1-beta**2)**2))
bb=s.symbols('B0:36');Bj=s.Matrix(6,6,bb);J=C19(Bj).jacobian(bb)
for bval in [0,s.Rational(1,2)]:
 eq('constraint_rank_beta_'+str(bval),J.subs(dict(zip(bb,list(B0.subs(beta,bval))))).rank(),19)
f=s.symbols('f0:16');be=s.symbols('dbeta');e=s.eye(4)+s.Matrix(4,4,f)
Bt=star*ex2(e)+(beta+be)*ex2(e)
T=s.Matrix(list(Bt)).jacobian((*f,be)).subs(dict.fromkeys((*f,be),0))
eq('coframe_plus_scalar_rank',T.subs(beta,s.Rational(1,2)).rank(),17)
eq('tangent_equals_relaxed_kernel',J.subs(dict(zip(bb,list(B0))))*T,s.zeros(21,17))
# Typed operational map: L_B(v)=1/2 B_IJ v^IJ, matrix B^T D.
LB=B0.T*D;Jem=s.simplify(LB*star*LB.inv())
eq('Maxwell_J_independent_of_beta',Jem,-star)
eq('Maxwell_J_wedge_selfadjoint',Jem.T*W,W*Jem)
eq('Maxwell_J_square',Jem*Jem,-s.eye(6))
# All 24 connection components. Build D Sigma at a normal coframe.
av=s.symbols('a0:24')
def A(mu,I,J):
 if I==J:return s.S.Zero
 return av[6*mu+P.index((I,J))]if I<J else -av[6*mu+P.index((J,I))]
torsion=[{ij:A(ij[0],I,ij[1])*eta[ij[1]]-A(ij[1],I,ij[0])*eta[ij[0]]for ij in P}for I in range(4)]
def coeff3(form2,one,tri):
 missing=next(i for i in range(4)if i not in tri)
 return sum(x*s.LeviCivita(*ij,one,missing)*s.LeviCivita(*tri,missing)for ij,x in form2.items())
DS=s.Matrix(6,4,lambda a,b:coeff3(torsion[P[a][0]],P[a][1],Q3[b])-
                  coeff3(torsion[P[a][1]],P[a][0],Q3[b]))
M=s.Matrix(list(DS)).jacobian(av);Minv=M.inv()
eq('D_Sigma_connection_rank',M.rank(),24)
grad=s.symbols('v0:4',real=True)
source=s.Matrix(6,4,lambda a,b:sum(grad[mu]*s.LeviCivita(mu,*P[a],next(i for i in range(4)if i not in Q3[b]))*
                s.LeviCivita(*Q3[b],next(i for i in range(4)if i not in Q3[b]))for mu in range(4)))
# (star+beta) D Sigma=-d beta wedge Sigma, inverted algebraically.
DSsol=s.simplify((star-beta*s.eye(6))*source/(1+beta**2))
Asol=s.simplify(Minv*s.Matrix(list(DSsol)))
eq('connection_equation_residual',(star+beta*s.eye(6))*DS.subs(dict(zip(av,list(Asol))))+source,s.zeros(6,4))
Cov=s.Matrix([(beta*((grad[J]*eta[J] if mu==I else 0)-(grad[I]*eta[I] if mu==J else 0))-
               sum(-s.LeviCivita(I,J,mu,L)*eta[mu]*eta[L]*eta[L]*grad[L]for L in range(4)))/(2*(1+beta**2))
              for mu in range(4)for I,J in P])
eq('covariant_contorsion_formula',Asol,Cov)
# Integrate B wedge dA by parts; on a stationary quadratic auxiliary action,
# L_on=1/2 L_linear=-1/2 d beta wedge Sigma^IJ wedge A_IJ.
linear=0
for I,J in P:
 for mu in range(4):
  for nu in range(4):
   linear-=2*eta[I]*eta[J]*grad[mu]*A(nu,I,J)*s.LeviCivita(mu,I,J,nu)
Lkin=s.factor((linear/2).subs(dict(zip(av,list(Asol)))))
v2=sum(eta[i]*grad[i]**2 for i in range(4))
eq('derived_scalar_kinetic_density',Lkin,-3*v2/(2*(1+beta**2)))
# The topological chart of the SAME family: B=(I+gamma star)Sigma.
gamma=s.symbols('gamma',real=True)
DStop=s.simplify(-(s.eye(6)-gamma*star)*star*source/(1+gamma**2))
Atop=s.simplify(Minv*s.Matrix(list(DStop)))
jtop=0
for a,(I,J) in enumerate(P):
 for b,(K,L) in enumerate(P):
  for mu in range(4):
   for nu in range(4):
    jtop-=2*eta[I]*eta[J]*grad[mu]*star[a,b]*A(nu,I,J)*s.LeviCivita(mu,K,L,nu)
topkin=s.factor((jtop/2).subs(dict(zip(av,list(Atop)))))
eq('topological_chart_kinetic_density',topkin,3*gamma*v2/(2*(1+gamma**2)))
Path(__file__).with_name('geometry_results.json').write_text(json.dumps({
 'checks':checks,'pair_basis':P,'connection_solution':[str(x)for x in Asol],
 'scalar_kinetic_density':str(Lkin),'topological_chart_kinetic_density':str(topkin),'softened_trace':str(tau),
 'scope':'Exact regular gravitational chart; no assertion for singular topological/degenerate strata.'},indent=2)+'\n')
print(len(checks),'geometry/connection checks passed')
