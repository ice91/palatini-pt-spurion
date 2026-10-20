"""Independent exact rational checks. No import from either derivation script.
Forms are dictionaries; ranks use Fraction Gaussian elimination, not SymPy.
"""
from pathlib import Path
from fractions import Fraction as Q
from itertools import combinations,permutations
import json

P=list(combinations(range(4),2));eta=[-1,1,1,1]; results={}
def eps(indices):
 if len(set(indices))<len(indices):return 0
 return (-1)**sum(indices[i]>indices[j] for i in range(len(indices)) for j in range(i+1,len(indices)))
def wedge(a,b):
 return sum(x*y*eps(ij+kl) for ij,x in a.items() for kl,y in b.items())
def form(v):return {ij:x for ij,x in zip(P,v) if x}
def star(B):
 return [[sum(-eps(ij+kl)*eta[kl[0]]*eta[kl[1]]*B[b][c]
              for b,kl in enumerate(P))for c in range(6)]for ij in P]
def simple(e):
 return [[e[I][mu]*e[J][nu]-e[I][nu]*e[J][mu] for mu,nu in P]for I,J in P]
def det(M):
 return sum(eps(p)*product(M[i][p[i]] for i in range(len(M)))for p in permutations(range(len(M))))
def product(seq):
 r=1
 for x in seq:r*=x
 return r
def C(B):
 G=[[wedge(form(B[a]),form(B[b]))for b in range(6)]for a in range(6)]
 v=sum(eps(P[a]+P[b])*G[a][b] for a in range(6) for b in range(6))*Q(1,6)
 return [G[a][b]-v*eps(P[a]+P[b]) for a in range(6) for b in range(a,6)]
def rank(M):
 A=[[Q(x)for x in row]for row in M];r=0
 for c in range(len(A[0])):
  p=next((i for i in range(r,len(A))if A[i][c]),None)
  if p is None:continue
  A[r],A[p]=A[p],A[r];v=A[r][c];A[r]=[x/v for x in A[r]]
  for i in range(len(A)):
   if i!=r:
    v=A[i][c];A[i]=[x-v*y for x,y in zip(A[i],A[r])]
  r+=1
  if r==len(A):break
 return r
def check(name,ok,**data):
 results[name]={'pass':bool(ok),**{k:str(v) for k,v in data.items()}}
 if not ok:raise AssertionError((name,data))
eye=[[int(i==j)for j in range(4)]for i in range(4)]
e=[[2,1,0,0],[0,1,1,0],[0,0,3,1],[0,0,0,2]]
for label,coframe in [('identity',eye),('nondiagonal',e)]:
 S=simple(coframe)
 for dual in [False,True]:
  B=star(S)if dual else S
  for sign in [-1,1]:
   Bsign=[[sign*x for x in row]for row in B]
   check('four_sector_'+label+'_'+str(dual)+'_'+str(sign),not any(C(Bsign)))
 B=star(S); columns=[]
 for n in range(36):
  Bp=[row[:]for row in B];Bm=[row[:]for row in B]
  i,j=divmod(n,6);Bp[i][j]+=1;Bm[i][j]-=1
  columns.append([(a-b)/2 for a,b in zip(C(Bp),C(Bm))])
 J=[list(row)for row in zip(*columns)]
 check('rational_constraint_rank_'+label,rank(J)==20,rank=rank(J))
 integrand=2*sum(eta[I]*eta[J]*wedge(form(B[a]),form(S[a]))for a,(I,J)in enumerate(P))
 check('Palatini_constant_curvature_'+label,integrand==12*det(coframe),BF_density=integrand,det_e=det(coframe))
# Natural real PT pullback keeps any constant two-form even; does not enforce C=0.
B=star(simple(eye));B[0][0]+=1
check('PT_even_BF_counterexample_not_simple',any(C(B)),constraint_residual=C(B))
# Constraint locus is not closed under addition, hence not a linear-projector image.
e2=[[2,0,0,0],[0,3,0,0],[0,0,1,0],[0,0,0,1]]
B1=star(simple(eye));B2=star(simple(e2));Bsum=[[x+y for x,y in zip(a,b)]for a,b in zip(B1,B2)]
check('two_simple_forms_sum_nonsimple',not any(C(B1))and not any(C(B2))and any(C(Bsum)),sum_residual=C(Bsum))
# Full Einstein symbol with a non-axis null direction: spatial momentum (3,4,0).
SP=[(i,j)for i in range(4)for j in range(i,4)]
def einstein(v,q):
 H=[[Q(0)for j in range(4)]for i in range(4)]
 for x,(i,j)in zip(v,SP):H[i][j]=H[j][i]=Q(x)
 qu=[eta[i]*q[i]for i in range(4)];q2=sum(q[i]*qu[i]for i in range(4))
 tr=sum(eta[i]*H[i][i]for i in range(4));qh=[sum(qu[a]*H[a][j]for a in range(4))for j in range(4)]
 qhq=sum(qu[j]*qh[j]for j in range(4))
 return [(-q[i]*qh[j]-q[j]*qh[i]+q2*H[i][j]+q[i]*q[j]*tr+
          (eta[i]*(qhq-q2*tr)if i==j else 0))/2 for i,j in SP]
def symbol(q):
 cols=[einstein([int(i==j)for i in range(10)],q)for j in range(10)]
 return [list(row)for row in zip(*cols)]
q=[-5,3,4,0];Dn=symbol(q)
gauge=[[q[i]*int(j==a)+q[j]*int(i==a)for i,j in SP]for a in range(4)]
u=[Q(-4,5),Q(3,5),Q(0)];v=[Q(0),Q(0),Q(1)]
plus=[u[i-1]*u[j-1]-v[i-1]*v[j-1]if i and j else Q(0)for i,j in SP]
cross=[u[i-1]*v[j-1]+v[i-1]*u[j-1]if i and j else Q(0)for i,j in SP]
check('rational_null_symbol_rank',rank(Dn)==4,rank=rank(Dn))
check('rational_offshell_symbol_rank',rank(symbol([-6,3,4,0]))==6)
check('oblique_two_physical_solutions',not any(einstein(plus,q))and not any(einstein(cross,q)))
span=[list(row)for row in zip(*(gauge+[plus,cross]))]
check('oblique_gauge_and_TT_span',rank(span)==6 and rank([list(row)for row in zip(*gauge)])==4)
# Independent pairing of the oblique physical polarizations.
def pairing(a,b):
 return sum((1 if i==j else 2)*a[n]*b[n]for n,(i,j)in enumerate(SP)if i and j)
check('oblique_physical_pairing',pairing(plus,plus)==2 and pairing(cross,cross)==2 and
      pairing(plus,cross)==0,plus_norm=pairing(plus,plus),cross_norm=pairing(cross,cross))
Path(__file__).with_name('independent_results.json').write_text(json.dumps({'checks':results,
 'method':'Pure Fraction elimination and independent exterior-form dictionaries; no SymPy imports.'},indent=2)+'\n')
print(len(results),'independent exact check cases passed')
