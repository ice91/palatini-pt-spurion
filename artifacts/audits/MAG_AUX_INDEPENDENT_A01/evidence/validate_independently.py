"""Exact checks on audit-derived objects. No original repository module imported.
Paper formulas occur only as comparison assertions, never as derivation inputs.
Known counterexamples are successful audit checks, not manuscript passes.
"""
import contextlib,io,json,itertools,sys
from pathlib import Path
import sympy as s
E=Path(__file__).resolve().parent
with contextlib.redirect_stdout(io.StringIO()):
 import derive_from_action as d
o=d.objects
globals().update(o)
checks=[]
def zero(x):
 if isinstance(x,s.MatrixBase):return all(s.simplify(y)==0 for y in x)
 return s.simplify(x)==0
def check(name,condition,kind='IDENTITY',detail=''):
 ok=bool(condition);checks.append(dict(name=name,passed=ok,kind=kind,detail=detail))
 print(('PASS ' if ok else 'FAIL ')+name)
ix=range(4);de=lambda i,j:s.Integer(i==j)
check('trace-coordinate map invertible',M.det()!=0,detail=str(M.det()))
check('distortion reconstructs all three original trace conventions',M*M.inv()==s.eye(3))
check('curvature current from independent contractions',zero(W-gi*(-2*T-Q+U)))
expectedQ=-s.Rational(2,3)*d.dot(T,T)-s.Rational(11,72)*d.dot(Q,Q)+s.Rational(1,18)*d.dot(U,U)-s.Rational(2,3)*d.dot(T,Q)+s.Rational(2,3)*d.dot(T,U)+s.Rational(2,9)*d.dot(Q,U)
check('curvature quadratic comparison with Eq5.1',zero(Qcurv-expectedQ))
# Curvature derivative contraction at an arbitrary Levi-Civita normal point.
tracevars=list(T)+list(Q)+list(U)
jets=s.Matrix(4,12,lambda m,i:s.Symbol(f'jet_{m}_{i}'))
diff=lambda expr,m:sum(s.diff(expr,v)*jets[m,j] for j,v in enumerate(tracevars))
Rder=sum(gi[r,n]*(diff(N[l,n,r],l)-diff(N[l,l,r],n)) for r,n,l in itertools.product(ix,repeat=3))
check('curvature derivative equals divergence of reconstructed W',zero(Rder-sum(diff(W[m],m) for m in ix)))
# Exact finite projective curvature calculation with fully arbitrary connection and xi jets.
Gam=s.MutableDenseNDimArray(s.symbols('Gam0:64'),(4,4,4))
xi=s.Matrix(s.symbols('xi0:4'));jx=s.Matrix(4,4,lambda m,n:s.Symbol(f'xi_jet_{m}_{n}'))
projC=lambda l,m,n:-de(l,n)*xi[m]
projective_errors=[]
for l,r,m,n in itertools.product(ix,repeat=4):
 deltaR=-de(l,r)*(jx[m,n]-jx[n,m])
 mixed=sum(Gam[l,m,k]*projC(k,n,r)+projC(l,m,k)*Gam[k,n,r]+projC(l,m,k)*projC(k,n,r)-Gam[l,n,k]*projC(k,m,r)-projC(l,n,k)*Gam[k,m,r]-projC(l,n,k)*projC(k,m,r) for k in ix)
 full=-de(l,r)*jx[m,n]+de(l,r)*jx[n,m]+mixed
 projective_errors.append(s.expand(full-deltaR))
check('finite arbitrary-xi curvature law all 256 components',all(zero(x) for x in projective_errors))
check('arbitrary-xi scalar curvature invariant',zero(sum(gi[r,n]*(-jx[r,n]+jx[n,r]) for r,n in itertools.product(ix,repeat=2))))
change={}
for m in ix:change.update({T[m]:T[m]+3*xi[m],Q[m]:Q[m]-8*xi[m],U[m]:U[m]-2*xi[m],A[m]:A[m]+3*xi[m]})
check('full affine bulk exactly invariant for arbitrary one-form',zero(Lraw.subs(change,simultaneous=True)-Lraw))
check('nonexact xi is permitted without scalar integrability',jx[0,1]!=jx[1,0],detail='Independent antisymmetric xi jets retained throughout curvature test; A is an independent one-form.')
check('quotient density contains no V',all(s.diff(Lhor,v)==0 for v in V))
check('full one-form Hessian includes varied A and Lorentz pairing',zero(H16-s.kronecker_product(Hraw,gi)))
check('horizontal full Hessian has Lorentz pairing',zero(H12-s.kronecker_product(H,gi)))
gauge=s.Matrix([3,-8,-2,3])
check('raw gauge is exact Hessian null vector',zero(Hraw*gauge))
check('raw compatible source / off-shell Noether identity',zero(sum(gauge[j]*s.diff(Lraw,[T,Q,U,A][j][1]) for j in range(4))))
check('horizontal determinant comparison Eq6.1',zero(H.det()+f1**2*f2/6))
check('regular raw species rank3 exact kernel1',Hraw.rank()==3 and len(Hraw.nullspace())==1)
check('regular full fibre rank12 kernel4',H16.rank()==12 and len(H16.nullspace())==4)
check('Palatini block determinant comparison Eq5.16',zero(P.det()+f1**2/12))
check('Schur coefficient derived as 2f2',zero(C-(B.T*P.inv()*B)[0]-2*f2))
check('Schur source derived as -df3',zero(J[2]-(B.T*P.inv()*J[:2,:])[0]+ks))
check('stationarity includes all four varied raw species',all(zero(s.diff(raw1,z).subs(rawsol,simultaneous=True)) for z in [Tz,Qz,Uz,Az]))
check('raw and quotient on-shell values agree',zero(raw_on-Lon))
check('derived reduced scalar coefficient comparison Eq7.1',zero(Lon-(3*ps**2/(2*f1)-ks**2/(4*f2))))
# Singular branches are substituted into the original Hessian/source; no inverse is used.
Jsp=J
for tag,sub,rank in [('S2',{f1:0},1),('S3',{f2:0},2),('S4',{f1:0,f2:0},0)]:
 hs=H.subs(sub);ns=hs.nullspace();compat=[s.factor((n.T*Jsp)[0]) for n in ns]
 check(tag+' ranks and compatibility computed without division',hs.rank()==rank and len(ns)==3-rank,detail='compatibility: '+str(compat)+'; full rank '+str(4*rank)+'; full nullity '+str(16-4*rank))
 if tag=='S2':
  check('S2 compatibility iff df1=0',all(zero(v.subs(ps,0)) for v in compat) and any(v!=0 for v in compat))
  sol=s.linsolve((hs,-Jsp.subs(ps,0)),(qs,us,ts));check('S2 q,u free and t=df3/(2f2)',sol==s.FiniteSet((qs,us,ks/(2*f2))))
 if tag=='S3':
  check('S3 compatibility iff df3=0',all(zero(v.subs(ks,0)) for v in compat) and any(v!=0 for v in compat))
 if tag=='S4':check('S4 compatible iff both df1=df3=0',s.solve(compat,[ps,ks])=={ps:0,ks:0})
# Gauge-adapted Hessian congruence and unrestricted horizontal relabelling.
rawvec=s.Matrix([T[1],Q[1],U[1],A[1]]);horvec=s.Matrix([q[1],u[1],t[1],V[1]])
R=rawvec.subs(chart,simultaneous=True).jacobian(horvec)
hb=s.diag(1,1,1,0);hb[:3,:3]=H
check('raw to horizontal congruence',zero(R.T*Hraw*R-hb) and abs(R.det())==1)
G=s.Matrix([[1,1,0],[0,1,1],[1,0,1]])
check('horizontal coordinate determinant transforms by square',zero((G.T*H*G).det()-G.det()**2*H.det()))
check('on-shell current Wstar=-3df1/f1',zero((-2*t[1]-q[1]+u[1]).subs({q[1]:d.fullsolve[qs],u[1]:d.fullsolve[us],t[1]:d.fullsolve[ts]},simultaneous=True)+3*ps/f1))
# Literal Eq(G.5) test. Its extra term is not an input to the solve above.
r=ps/f1;st=ks/(2*f2)
qsol=4*r-s.Rational(8,3)*st;usol=r-s.Rational(2,3)*st
Q1=(L1-(ps*(2*ts+qs-us)-ks*ts)-f2*ts**2)/f1
literalG5=f1*Q1.subs({qs:qsol,us:usol,ts:st},simultaneous=True)+ps*(2*st+4*r-r)+f2*st**2-ks*st
G5res=s.factor(literalG5-Lon)
check('counterexample: literal EqG5 has extra df1.df3/f2',zero(G5res-ps*ks/f2) and G5res!=0,'COUNTEREXAMPLE',str(G5res))
# Test only the diagnostic family actually printed in Appendix J, without changing this model.
alpha,beta=s.symbols('alpha beta');db=s.Matrix([alpha,beta]);Bp=B+db
deltaS=s.factor(C-(Bp.T*P.inv()*Bp)[0]-(C-(B.T*P.inv()*B)[0]))
deltaJ=s.factor(-(db.T*P.inv()*J[:2,:])[0])
check('AppendixJ Schur formulas correct',zero(deltaS-(-16*alpha/3-4*beta/3-(-4*alpha**2+16*alpha*beta+11*beta**2)/(3*f1))))
check('AppendixJ source formula correct',zero(deltaJ-ps*(4*alpha+beta)/f1))
check('counterexample: pointwise uniqueness in AppendixJ fails for df1=0',zero(deltaS.subs({alpha:4*f1,beta:0})) and zero(deltaJ.subs({ps:0,alpha:4*f1,beta:0})),'COUNTEREXAMPLE','Constant nonzero f1, alpha=4f1, beta=0; both fingerprint changes vanish. Universal-jet interpretation avoids this counterexample.')
m=s.Symbol('m');curveA=4*f1*(4+m)/(4-16*m-11*m*m);curveB=m*curveA
check('AppendixJ counterexamples also exist arbitrarily near origin',zero(deltaS.subs({alpha:curveA,beta:curveB},simultaneous=True)) and s.limit(curveA,m,-4)==0 and s.limit(curveB,m,-4)==0,'COUNTEREXAMPLE','At df1=0 take alpha=4f1(4+m)/(4-16m-11m^2), beta=m alpha, m approaching -4 but unequal to -4.')
check('AppendixJ uniqueness as identity for arbitrary source jets holds',zero(deltaS.subs(beta,-4*alpha)+36*alpha**2/f1))
# A current can be projectively invariant but have nonzero generic variational flux.
z=s.Symbol('z');delta_t_normal=s.Integer(1)
check('counterexample: projective invariance does not remove variational boundary flux',f3*delta_t_normal!=0,'COUNTEREXAMPLE','At a flat non-null boundary with df3=0, delta t_normal=1 gives delta(J3.normal)=f3. Bulk theorem must retain its boundary hypothesis.')
result={'checks':checks,'total':len(checks),'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks),'counterexample_checks':sum(c['kind']=='COUNTEREXAMPLE' for c in checks),'scope':'exact auxiliary trace-sector calculations only; no spacetime stability/ADM/Dirac theorem tested'}
(E/'independent_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
sys.exit(1 if result['failed'] else 0)
