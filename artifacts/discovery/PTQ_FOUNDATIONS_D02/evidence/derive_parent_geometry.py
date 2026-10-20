"""NEW D02 fixed parent: full compatible connection varied before any torsion ansatz."""
from pathlib import Path
from itertools import combinations
import json
import sympy as s
HERE=Path(__file__).resolve().parent
sig=[-1,1,1,1]; pairs=list(combinations(range(4),2))
slots=[(a,b,mu) for a,b in pairs for mu in range(4)]
w=s.symbols('w0:24'); M2,beta=s.symbols('M2 beta',nonzero=True,real=True)
v=s.symbols('v0:4',real=True)
def low(a,b,mu):
 if a==b:return s.Integer(0)
 return w[slots.index((a,b,mu))] if a<b else -w[slots.index((b,a,mu))]
def W(a,b,mu):return sig[a]*low(a,b,mu)
def T(a,mu,nu):return W(a,nu,mu)-W(a,mu,nu)
trace=s.Matrix([sum(T(a,mu,a) for a in range(4)) for mu in range(4)])
Rquad=s.expand(sum(sig[b]*(W(a,c,a)*W(c,b,b)-W(a,c,b)*W(c,b,a)) for a in range(4) for b in range(4) for c in range(4)))
L=s.expand(M2*Rquad/2+beta*sum(sig[mu]*trace[mu]*v[mu] for mu in range(4)))
H0=s.hessian(Rquad/2,w)
J=s.Matrix([s.diff(sum(sig[mu]*trace[mu]*v[mu] for mu in range(4)),x) for x in w])
assert H0.rank()==24
sol=s.simplify(-beta/M2*H0.inv()*J)
sub=dict(zip(w,sol)); tr=s.simplify(trace.subs(sub))
checks={}
def zero(name,expr):
 terms=list(expr) if isinstance(expr,s.MatrixBase) else [expr]
 checks[name]=all(s.simplify(x)==0 for x in terms)
 assert checks[name],name
zero('full_stationary_24_equations',s.Matrix([s.diff(L,x) for x in w]).subs(sub))
zero('derived_trace',tr-s.Rational(3,2)*beta/M2*s.Matrix(v))
# Test the result, not an input ansatz, against pure trace reconstructed from the computed trace.
res=[]
for a in range(4):
 for mu in range(4):
  for nu in range(4):
   pred=((1 if a==nu else 0)*tr[mu]-(1 if a==mu else 0)*tr[nu])/3
   res.append(s.simplify(T(a,mu,nu).subs(sub)-pred))
zero('no_axial_or_tensor_torsion_in_solution',s.Matrix(res))
v2=sum(sig[mu]*v[mu]**2 for mu in range(4))
Lstar=s.factor(L.subs(sub))
zero('stationary_correction',Lstar-s.Rational(3,4)*beta**2/M2*v2)
# Independent raw tensor contraction of torsion norm at the derived solution.
Tnorm=s.expand(sum(sig[a]*sig[mu]*sig[nu]*T(a,mu,nu).subs(sub)**2 for a in range(4) for mu in range(4) for nu in range(4)))
zero('pure_trace_norm_from_components',Tnorm-s.Rational(2,3)*sum(sig[i]*tr[i]**2 for i in range(4)))
# Verify the derivative current in R for pure trace using an arbitrary jet, after the solve.
jet=s.Matrix(4,4,lambda i,j:s.Symbol('dT_%d_%d'%(i,j),real=True))
def dK(a,b,nu,mu):
 # K^a_bnu=(g_bnu t^a-delta^a_nu t_b)/3, derivative with respect to x^mu.
 return ((sig[b] if b==nu else 0)*sig[a]*jet[mu,a]-(1 if a==nu else 0)*jet[mu,b])/3
Rder=s.expand(sum(sig[b]*(dK(a,b,b,a)-dK(a,b,a,b)) for a in range(4) for b in range(4)))
zero('trace_derivative_curvature_is_plus_two_divergence',Rder-2*sum(sig[i]*jet[i,i] for i in range(4)))
# For an arbitrary compatible contorsion (before solving), the derivative current is 2t.
zero('arbitrary_contorsion_first_curvature_current',s.Matrix([sum(sig[b]*W(a,b,b) for b in range(4))-sig[a]*trace[a] for a in range(4)]))
zero('arbitrary_contorsion_second_curvature_current',s.Matrix([sum(W(a,b,a) for a in range(4))+trace[b] for b in range(4)]))
Ltrace=trace.jacobian(w)
zero('actual_auxiliary_Schur_gradient_block',Ltrace*H0.inv()*Ltrace.T+s.Rational(3,2)*s.diag(*sig))
result={'sympy':s.__version__,'conventions':{'signature':sig,'torsion':'T^a_mn=omega^a_n m-omega^a_m n','trace':'t_m=T^a_m a','R':'g^bn R^a_b a n'},'H0_rank':H0.rank(),'H0_det':str(H0.det()),'H0':[[str(x) for x in row] for row in H0.tolist()],'trace_stationary':[str(x) for x in tr],'connection_solution':{str(slots[i]):str(sol[i]) for i in range(24)},'stationary_density':str(Lstar),'torsion_norm':str(s.factor(Tnorm)),'checks':checks,'all_pass':all(checks.values())}
(HERE/'geometry_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:val for k,val in result.items() if k not in ['H0','connection_solution']},indent=2))
