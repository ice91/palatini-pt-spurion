"""Independent second-cycle check: Fraction exterior forms, no SymPy or first-script imports."""
from fractions import Fraction as F
from itertools import combinations,permutations
from pathlib import Path
import json
OUT=Path(__file__).resolve().parent
pairs=list(combinations(range(4),2)); eta=[-1,1,1,1]; checks={}
def check(name,c):
    checks[name]=bool(c)
    if not c: raise AssertionError(name)
def sign(indices):
    if len(set(indices))!=len(indices): return 0
    return (-1)**sum(indices[i]>indices[j] for i in range(len(indices)) for j in range(i+1,len(indices)))
def wedge(a,b):
    out={}
    for ia,va in a.items():
        for ib,vb in b.items():
            ij=ia+ib; sg=sign(ij)
            if sg:
                key=tuple(sorted(ij)); out[key]=out.get(key,F(0))+sg*va*vb
    return {i:v for i,v in out.items() if v}
def add(a,b,wa=F(1),wb=F(1)):
    return {k:wa*a.get(k,0)+wb*b.get(k,0) for k in set(a)|set(b) if wa*a.get(k,0)+wb*b.get(k,0)}
def B(e):
    oneforms=[{(j,):F(e[i][j]) for j in range(4) if e[i][j]} for i in range(4)]
    sigma=[wedge(oneforms[i],oneforms[j]) for i,j in pairs]
    b=[]
    for i,j in pairs:
        out={}
        for n,(k,l) in enumerate(pairs):
            sg=eta[i]*eta[j]*sign((i,j,k,l))
            if sg: out=add(out,sigma[n],F(1),F(sg))
        b.append(out)
    return b
def diag(vals):return [[F(vals[i]) if i==j else F(0) for j in range(4)] for i in range(4)]
W=[[F(sign(a+b)) for b in pairs] for a in pairs]
D=[-1,-1,-1,1,1,1]
def gram(b):return [[wedge(a,c).get((0,1,2,3),F(0)) for c in b] for a in b]
def info(b):
    G=gram(b); v=sum(W[i][j]*G[i][j] for i in range(6) for j in range(6))/6
    Q=2*sum(D[i]*G[i][i] for i in range(6))
    C=[[G[i][j]-v*W[i][j]-(Q*D[i]/12 if i==j else 0) for j in range(6)] for i in range(6)]
    CW=[[sum(C[i][k]*W[k][j] for k in range(6)) for j in range(6)] for i in range(6)]
    invariant=sum(CW[i][j]*CW[j][i] for i in range(6) for j in range(6))/v**2
    return G,v,Q,C,invariant
b1=B(diag([1,1,1,1])); b2=B(diag([2,3,1,1])); b3=B(diag([2,1,3,1]))
for n,b in enumerate([b1,b2,b3]):
    G,v,Q,C,I=info(b)
    check(f'hard_micro_{n}',Q==0 and not any(map(any,C)))
rows=[]
for weight in [F(0),F(1,4),F(1,3),F(1,2),F(3,4),F(1)]:
    mean=[add(a,b,1-weight,weight) for a,b in zip(b1,b2)]
    G,v,Q,C,I=info(mean)
    check(f'Q_zero_weight_{weight}',Q==0)
    check(f'residual_iff_nontrivial_average_{weight}',any(map(any,C))==(0<weight<1))
    if weight==F(1,2):
        halfC=C
        check('half_v',v==-F(19,6))
        check('half_C_03_12',C[2][3]==F(1,6))
        check('half_C_01_23',C[0][5]==-F(1,3))
        check('half_C_02_13',C[1][4]==-F(1,6))
        check('half_invariant',I==F(12,361))
    rows.append({'weight':str(weight),'v':str(v),'Q':str(Q),'I19':str(I),'nonzero_C19':any(map(any,C))})
# An independent two-normal-direction witness, using an exact 2x2 minor.
other=info([add(a,b,F(1,2),F(1,2)) for a,b in zip(b1,b3)])[3]
flat1=[v for row in halfC for v in row]; flat2=[v for row in other for v in row]
minor=next((flat1[i]*flat2[j]-flat1[j]*flat2[i] for i,j in combinations(range(36),2) if flat1[i]*flat2[j]-flat1[j]*flat2[i]),None)
check('two_independent_non_scalar_normals',minor is not None)
(OUT/'independent_results.json').write_text(json.dumps({'checks':checks,'number_passed':len(checks),'weight_witnesses':rows,'nonzero_two_direction_minor':str(minor),'method':'Fraction one-form wedge expansion; no import or result read from derive_closure.py'},indent=2)+'\n')
print(json.dumps({'checks_passed':len(checks),'weights_checked':len(rows),'nonzero_minor':str(minor)},indent=2))
