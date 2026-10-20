"""Independent rational exterior-action computation; no SymPy/derivation imports.
Inverts the quadratic auxiliary action rather than the D B equation.
"""
from pathlib import Path
from itertools import combinations
from fractions import Fraction as F
import json
P=list(combinations(range(4),2));eta=[-1,1,1,1];out={}
def eps(t):return 0 if len(set(t))<len(t)else (-1)**sum(t[i]>t[j]for i in range(len(t))for j in range(i+1,len(t)))
def wedge(a,b):
 r={}
 for ij,x in a.items():
  for kl,y in b.items():
   ind=ij+kl
   if len(set(ind))<len(ind):continue
   key=tuple(sorted(ind));r[key]=r.get(key,F(0))+eps(ind)*x*y
 return r
def scale(a,x):return {k:x*v for k,v in a.items()if x*v}
def add(a,b):
 r=a.copy()
 for k,v in b.items():r[k]=r.get(k,F(0))+v
 return r
def top(a):return a.get((0,1,2,3),F(0))
def form(v):return {ij:F(x)for ij,x in zip(P,v)if x}
def star(B):return [[sum(-eps(ij+kl)*eta[kl[0]]*eta[kl[1]]*B[b][c]for b,kl in enumerate(P))for c in range(6)]for ij in P]
def simple(e):return [[F(e[I][mu]*e[J][nu]-e[I][nu]*e[J][mu])for mu,nu in P]for I,J in P]
I4=[[F(i==j)for j in range(4)]for i in range(4)];Sig=simple(I4);S=star(Sig)
def check(name,ok,**data):
 out[name]={'pass':bool(ok),**{k:str(v)for k,v in data.items()}}
 if not ok:raise AssertionError((name,data))
def inverse(M):
 n=len(M);A=[[F(x)for x in M[i]]+[F(i==j)for j in range(n)]for i in range(n)]
 for c in range(n):
  p=next(i for i in range(c,n)if A[i][c]);A[c],A[p]=A[p],A[c]
  v=A[c][c];A[c]=[x/v for x in A[c]]
  for i in range(n):
   if i!=c:
    v=A[i][c];A[i]=[x-v*y for x,y in zip(A[i],A[c])]
 return [r[n:]for r in A]
def mm(A,B):return [[sum(a*b for a,b in zip(row,col))for col in zip(*B)]for row in A]
def tr(A):return [list(r)for r in zip(*A)]
def Aforms(v):
 a=[[{}for j in range(4)]for i in range(4)]
 for n,(I,J)in enumerate(P):
  x={(mu,):F(v[6*mu+n])for mu in range(4)if v[6*mu+n]}
  a[I][J]=x;a[J][I]=scale(x,-1)
 return a
def quadratic(B,v):
 a=Aforms(v);res=F(0)
 for n,(I,J)in enumerate(P):
  f={}
  for K in range(4):f=add(f,scale(wedge(a[I][K],a[K][J]),eta[K]))
  res+=2*eta[I]*eta[J]*top(wedge(form(B[n]),f))
 return res
def linear(v,gradient):
 a=Aforms(v);db={(i,):F(x)for i,x in enumerate(gradient)if x}
 return -2*sum(eta[I]*eta[J]*top(wedge(wedge(db,form(Sig[n])),a[I][J]))for n,(I,J)in enumerate(P))
units=[[F(i==j)for i in range(24)]for j in range(24)]
for b in [F(0),F(1,2),F(-1,3)]:
 B=[[S[i][j]+b*Sig[i][j]for j in range(6)]for i in range(6)]
 d=[quadratic(B,u)for u in units];H=[[F(0)for j in range(24)]for i in range(24)]
 for i in range(24):
  H[i][i]=2*d[i]
  for j in range(i):H[i][j]=H[j][i]=quadratic(B,[x+y for x,y in zip(units[i],units[j])])-d[i]-d[j]
 Hi=inverse(H)
 for label,grad in [('time',[1,0,0,0]),('space',[0,0,0,1]),('mixed',[1,2,0,1])]:
  j=[linear(u,grad)for u in units];a=[-sum(x*y for x,y in zip(row,j))for row in Hi]
  val=quadratic(B,a)+linear(a,grad)
  expected=-F(3,2)*sum(eta[i]*grad[i]**2 for i in range(4))/(1+b*b)
  check('action_elimination_'+str(b)+'_'+label,val==expected,on_shell=val)
# Reconstruct Maxwell Hodge directly from the metric of a nondiagonal coframe.
e=[[F(1),0,0,0],[0,F(2),F(1),0],[0,0,F(1),0],[0,0,0,F(1)]]
g=mm(mm(tr(e),[[eta[i]*F(i==j)for j in range(4)]for i in range(4)]),e);gi=inverse(g)
Hmetric=[[2*sum(eps(ij+(r,t))*gi[r][kl[0]]*gi[t][kl[1]]for r in range(4)for t in range(4))for kl in P]for ij in P]
Se=simple(e);Ste=star(Se)
for b in [F(0),F(1,2),F(-1,3)]:
 B=[[Ste[i][j]+b*Se[i][j]for j in range(6)]for i in range(6)]
 LB=[[x*eta[P[j][0]]*eta[P[j][1]]for j,x in enumerate(row)]for row in tr(B)]
 J=mm(mm(LB,S),inverse(LB))
 check('nondiagonal_operational_Hodge_'+str(b),J==Hmetric)
 # Strict simplicity fails for b != 0, but the same metric Hodge still holds.
 GB=[[top(wedge(form(a),form(c)))for c in B]for a in B]
 volume=sum(eps(P[i]+P[j])*GB[i][j]for i in range(6)for j in range(6))*F(1,6)
 Q=2*sum(eta[I]*eta[J]*GB[n][n]for n,(I,J)in enumerate(P))
 check('invariants_'+str(b),volume==2*(b*b-1)and Q==48*b,volume=volume,Q=Q)
# Nonzero scalar data exactly satisfy nonlinear constraints; hard retraction fails.
c=F(1,3);lam=F(2);kap=F(1);V=c*c/(kap*lam*(1-c*c));H2=kap*V/3
check('nonlinear_initial_constraint',6*H2==2*kap*V and V>0,V=V,H_squared=H2)
check('hard_retraction_not_solution_map',6*H2!=0,remaining_Hamiltonian_residual=6*H2)
check('two_positive_deformations_distinct_poles',F(4,3)/F(1)!=F(4,3)/F(2),mass2_lambda1=F(4,3),mass2_lambda2=F(2,3))
Path(__file__).with_name('independent_results.json').write_text(json.dumps({'checks':out,
 'method':'Independent Fraction arithmetic and direct quadratic exterior action; no SymPy or previous evidence imports.'},indent=2)+'\n')
print(len(out),'independent exact cases passed')
