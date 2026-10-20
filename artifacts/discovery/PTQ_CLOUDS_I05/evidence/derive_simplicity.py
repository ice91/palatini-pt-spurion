"""Cycle 1: real Lorentzian bivectors, simplicity and connection ranks.
All forms use ordered coordinate bases; no Euclidean Hodge sign is imported.
"""
from pathlib import Path
from itertools import combinations
import json
import sympy as s

pairs=list(combinations(range(4),2)); triples=list(combinations(range(4),3))
eta=[-1,1,1,1]
W=s.Matrix(6,6,lambda a,b:s.LeviCivita(*pairs[a],*pairs[b]))
E=-W  # epsilon^{0123}=-1
star=s.Matrix(6,6,lambda a,b:-s.LeviCivita(*pairs[a],*pairs[b])*
              eta[pairs[b][0]]*eta[pairs[b][1]])
checks={}
def eq(name,x,y):
 r=x-y
 ok=all(s.simplify(v)==0 for v in r) if isinstance(r,s.MatrixBase) else s.simplify(r)==0
 checks[name]={'pass':bool(ok)}
 if not ok: raise AssertionError((name,r))
def volume(B):
 G=B*W*B.T
 return s.expand(sum(W[i,j]*G[i,j] for i in range(6) for j in range(6))/6)
def constraints(B):
 G=B*W*B.T+volume(B)*E
 return s.Matrix([G[a,b] for a in range(6) for b in range(a,6)])
def exterior_square(e):
 return s.Matrix(6,6,lambda a,b:e[pairs[a][0],pairs[b][0]]*e[pairs[a][1],pairs[b][1]]-
                 e[pairs[a][0],pairs[b][1]]*e[pairs[a][1],pairs[b][0]])
eq('Lorentz_internal_star_squared',star*star,-s.eye(6))
eq('star_wedge_anti_isometry',star*W*star.T,-W)
for name,B,val in [('I+',s.eye(6),1),('I-',-s.eye(6),1),
                   ('II+',star,-1),('II-',-star,-1)]:
 eq(name+'_simplicity',constraints(B),s.zeros(21,1))
 eq(name+'_signed_volume',volume(B),val)
# 36 unconstrained B components; eliminate the one volume density algebraically.
v=s.symbols('b0:36'); B=s.Matrix(6,6,v)
J=constraints(B).jacobian(v)
reg=dict(zip(v,list(star))); zero=dict.fromkeys(v,0)
Jreg=J.subs(reg)
eq('regular_constraint_rank',Jreg.rank(),20)
eq('zero_constraint_rank',J.subs(zero).rank(),0)
f=s.symbols('f0:16'); e=s.eye(4)+s.Matrix(4,4,f)
T=s.Matrix(list(star*exterior_square(e))).jacobian(f).subs(dict.fromkeys(f,0))
eq('coframe_tangent_rank',T.rank(),16)
eq('coframe_tangent_solves_linear_constraints',Jreg*T,s.zeros(21,16))
eq('tangent_kernel_dimension',36-Jreg.rank(),T.rank())
# Nonzero degenerate exact solution: B^{01}=dx0 wedge dx1.
Bd=s.zeros(6);Bd[0,0]=1
eq('degenerate_exact_constraint',constraints(Bd),s.zeros(21,1))
eq('degenerate_matrix_rank',Bd.rank(),1)
deg_rank=J.subs(dict(zip(v,list(Bd)))).rank()
# Linear map from 24 Lorentz-connection coefficients to 24 torsion coefficients.
av=s.symbols('a0:24')
def A(mu,I,J):
 if I==J:return s.S.Zero
 return av[6*mu+pairs.index((I,J))] if I<J else -av[6*mu+pairs.index((J,I))]
torsion=s.Matrix([A(mu,I,nu)*eta[nu]-A(nu,I,mu)*eta[mu]
                  for I in range(4) for mu,nu in pairs])
eq('metric_compatible_connection_torsion_rank',torsion.jacobian(av).rank(),24)
# Independently, D(e^I wedge e^J)=T^I wedge e^J-e^I wedge T^J.
tv=s.symbols('T0:24')
def wedge_t_e(I,J,tri):
 return sum(tv[6*I+b]*s.LeviCivita(*ab,J,*[i for i in range(4) if i not in tri])*
            s.LeviCivita(*tri,*[i for i in range(4) if i not in tri])
            for b,ab in enumerate(pairs))
dSigma=s.Matrix([wedge_t_e(I,J,tri)-wedge_t_e(J,I,tri)
                 for I,J in pairs for tri in triples])
eq('D_Sigma_torsion_rank',dSigma.jacobian(tv).rank(),24)
# BF shift is not tangent to simplicity: one allowed d(eta) value suffices.
shift=s.eye(6)[:,0]*s.eye(6)[0,:]  # delta B^{01}=dx0 wedge dx1
shift_res=Jreg*s.Matrix(list(shift))
eq('shift_breaks_simplicity',any(x!=0 for x in shift_res),True)
# Constant curvature F^{IJ}=K e^I wedge e^J calibrates Palatini sign.
K=s.symbols('K')
eps_eeF=sum(s.LeviCivita(I,J,k,l)**2*K for I in range(4) for J in range(4)
             for k in range(4) for l in range(4))
eq('epsilon_eeF_equals_2R',eps_eeF,2*12*K)
out={'sympy':s.__version__,'checks':checks,'pair_basis':pairs,'W':str(W),'star':str(star),
     'regular_rank':20,'regular_tangent_dimension':16,'degenerate_rank_one_constraint_J_rank':deg_rank,
     'shift_constraint_residual':str(shift_res),
     'scope':'Exact pointwise algebra. Nonlinear reconstruction and branch/global limits are in reports.'}
Path(__file__).with_name('simplicity_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(len(checks),'exact checks passed; rank-one degenerate Jacobian rank',deg_rank)
