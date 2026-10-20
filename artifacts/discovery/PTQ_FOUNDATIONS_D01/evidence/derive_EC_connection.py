"""Pointwise exact connection-equation rank derived from NEW Candidate 1 EC action."""
from pathlib import Path
from itertools import combinations,permutations
import sympy as s
import json
pairs=list(combinations(range(4),2)); triples=list(combinations(range(4),3))
slots=[(a,p) for a in range(4) for p in pairs]
unknown=s.symbols('T0:24'); eq=[]
def sign(word):return (-1)**sum(word[i]>word[j] for i in range(len(word)) for j in range(i+1,len(word)))
def eps(word):return sign(word) if len(set(word))==4 else 0
for a,b in pairs:
 for triple in triples:
  expr=0
  for c in range(4):
   for d in range(4):
    for p in pairs:
     word=(c,)+p
     if len(set(word))==3 and tuple(sorted(word))==triple:
      expr+=eps((a,b,c,d))*sign(word)*unknown[slots.index((d,p))]
  eq.append(expr)
A=s.linear_eq_to_matrix(eq,unknown)[0]
assert A.rank()==24
result={'equation':'epsilon_ABCD e^C wedge T^D=0 at a nondegenerate coframe','shape':list(A.shape),'rank':A.rank(),'determinant':str(A.det()),'conclusion':'all 24 torsion components vanish; no pure-trace assumption','rank_check_pass':True}
Path(__file__).with_name('EC_connection_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
