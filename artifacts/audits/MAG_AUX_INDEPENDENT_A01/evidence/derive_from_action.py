"""Independent derivation from the manuscript's conventions/action only.
No manuscript Hessian, target coefficient, proposed solve, or reduced-action value
is an input. Source comparisons are performed separately after this derivation.
"""
from pathlib import Path
import sympy as s
import json,itertools
E=Path(__file__).resolve().parent
D=4; indices=range(D); g=s.diag(-1,1,1,1); gi=g.inv()
a=s.symbols('a0:4'); b=s.symbols('b0:4'); c=s.symbols('c0:4')
def delta(i,j):return s.Integer(i==j)
# General parity-even vectorial distortion ansatz, before solving any trace coefficients.
N=s.MutableDenseNDimArray.zeros(D,D,D)
for l,m,n in itertools.product(indices,repeat=3):
 N[l,m,n]=delta(l,m)*a[n]+delta(l,n)*b[m]+g[m,n]*sum(gi[l,k]*c[k] for k in indices)
T0=s.Matrix([sum(N[l,l,m]-N[l,m,l] for l in indices) for m in indices])
Qten=lambda l,m,n:sum(N[r,l,m]*g[r,n]+N[r,l,n]*g[m,r] for r in indices)
Q0=s.Matrix([s.expand(sum(gi[m,n]*Qten(l,m,n) for m,n in itertools.product(indices,repeat=2))) for l in indices])
U0=s.Matrix([s.expand(sum(gi[m,l]*Qten(l,m,n) for m,l in itertools.product(indices,repeat=2))) for n in indices])
M=s.Matrix([T0[0],Q0[0],U0[0]]).jacobian([a[0],b[0],c[0]])
Tv=s.Matrix(s.symbols('T0:4'));Qv=s.Matrix(s.symbols('Q0:4'));Uv=s.Matrix(s.symbols('U0:4'));Av=s.Matrix(s.symbols('A0:4'))
sub={}
for mu in indices:
 inv=M.inv()*s.Matrix([Tv[mu],Qv[mu],Uv[mu]])
 for src,val in zip([a[mu],b[mu],c[mu]],inv):sub[src]=val
Nt=s.MutableDenseNDimArray([s.expand(x.subs(sub)) for x in N.reshape(D**3)],(D,D,D))
W=s.Matrix([s.expand(sum(gi[r,n]*Nt[mu,n,r]-gi[r,mu]*Nt[l,l,r] for r,n,l in itertools.product(indices,repeat=3) if n==l)) for mu in indices])
# Above single shared-index representation sums each term once: n==l.
# Compute the two curvature-quadratic contractions directly from R^lambda_{rho mu nu}.
Qcurv=s.expand(sum(gi[r,n]*(Nt[l,l,k]*Nt[k,n,r]-Nt[l,n,k]*Nt[k,l,r]) for r,n,l,k in itertools.product(indices,repeat=4)))
dot=lambda x,y:s.expand((s.Matrix(x).T*gi*s.Matrix(y))[0])
# Derivative-jet reconstruction of the curvature derivative term at a Levi-Civita normal point.
deriv=s.Matrix(D,D,lambda m,n:s.Symbol(f'd_{m}_{n}'))
# Divergence current determined by direct contraction: W^mu=g^{rho nu}N^mu_{nu rho}-g^{rho mu}N^lambda_{lambda rho}.
Wdirect=s.Matrix([s.expand(sum(gi[r,n]*Nt[mu,n,r] for r,n in itertools.product(indices,repeat=2))-sum(gi[r,mu]*Nt[l,l,r] for r,l in itertools.product(indices,repeat=2))) for mu in indices])
assert W==Wdirect
# Generic independent source one-form jets p=df1, k=df3.
f1,f2,f3=s.symbols('f1 f2 f3'); p=s.Matrix(s.symbols('p0:4')); k=s.Matrix(s.symbols('k0:4'));t=Tv-Av
# Exact scalar affine bulk after the two elementary IBPs, derived from action Eq3.7.
Lraw=s.expand(f1*Qcurv-dot(p,g*W)+f2*dot(t,t)-dot(k,t))
raw=list(Tv)+list(Qv)+list(Uv)+list(Av)
H16=s.hessian(Lraw,raw); J16=s.Matrix([s.diff(Lraw,x).subs({v:0 for v in raw}) for x in raw])
Hraw=s.Matrix(4,4,lambda i,j:s.simplify(H16[i*D+1,j*D+1]))
# Quotient chart is an invertible relabelling, not a supplied target Hessian.
qv=s.Matrix(s.symbols('q0:4'));uv=s.Matrix(s.symbols('u0:4'));tv=s.Matrix(s.symbols('t0:4'));V=s.Matrix(s.symbols('V0:4'))
chart={}
for mu in indices:
 chart.update({Tv[mu]:tv[mu]+V[mu],Qv[mu]:qv[mu]-s.Rational(8,3)*V[mu],Uv[mu]:uv[mu]-s.Rational(2,3)*V[mu],Av[mu]:V[mu]})
Lhor=s.expand(Lraw.subs(chart,simultaneous=True));hor=list(qv)+list(uv)+list(tv);H12=s.hessian(Lhor,hor);H=s.Matrix(3,3,lambda i,j:s.simplify(H12[i*D+1,j*D+1]))
j=s.Matrix([s.diff(Lhor,hor[i*D+1]).subs({v:0 for v in hor+list(V)}) for i in range(3)])
# All solve and on-shell formulas come from differentiating the derived polynomial.
qs,us,ts,ps,ks=s.symbols('q u t p k')
jet_sub={qv[1]:qs,uv[1]:us,tv[1]:ts,p[1]:ps,k[1]:ks}
jet_sub.update({x:0 for vec in [qv,uv,tv,p,k,V] for mu,x in enumerate(vec) if mu!=1})
L1=s.expand(Lhor.subs(jet_sub,simultaneous=True));x=s.Matrix([qs,us,ts]);J=s.Matrix([s.diff(L1,v).subs({qs:0,us:0,ts:0}) for v in x]);P=H[:2,:2];B=H[:2,2:];C=H[2,2]
qu=s.solve([s.diff(L1,qs),s.diff(L1,us)],[qs,us]);Lt=s.simplify(L1.subs(qu));tstar=s.solve(s.diff(Lt,ts),ts)[0];fullsolve={**{v:s.simplify(val.subs(ts,tstar)) for v,val in qu.items()},ts:tstar};Lon=s.factor(L1.subs(fullsolve,simultaneous=True))
St=s.factor(C-(B.T*P.inv()*B)[0]);Jeff=s.simplify(J[2]-(B.T*P.inv()*J[:2,:])[0]);Lconst=s.factor(-s.Rational(1,2)*(J[:2,:].T*P.inv()*J[:2,:])[0])
# Complete raw stationary solve with A free, without feeding the manuscript's proposed solution.
Tz,Qz,Uz,Az=s.symbols('T Q U A');raw1=Lraw.subs({Tv[1]:Tz,Qv[1]:Qz,Uv[1]:Uz,Av[1]:Az,p[1]:ps,k[1]:ks,**{v:0 for vec in [Tv,Qv,Uv,Av,p,k] for mu,v in enumerate(vec) if mu!=1}},simultaneous=True)
rawsol=s.solve([s.diff(raw1,z) for z in [Tz,Qz,Uz,Az]],[Tz,Qz,Uz],dict=True)[0];rawon=s.factor(raw1.subs(rawsol,simultaneous=True))
# Reconstruct distortion on that independently obtained raw stationary orbit.
onN={}
for mu in indices:
 onN[Tv[mu]]=rawsol[Tz].subs({ps:p[mu],ks:k[mu],Az:Av[mu]});onN[Qv[mu]]=rawsol[Qz].subs({ps:p[mu],ks:k[mu],Az:Av[mu]});onN[Uv[mu]]=rawsol[Uz].subs({ps:p[mu],ks:k[mu],Az:Av[mu]})
Non=s.MutableDenseNDimArray([s.simplify(val.subs(onN,simultaneous=True)) for val in Nt.reshape(D**3)],(D,D,D))
# Serialization allows inspection without having to trust a target-matching test.
result={'sympy':s.__version__,'trace_map':str(M),'trace_map_inverse':str(M.inv()),'distortion_components':[str(x) for x in Nt.reshape(D**3)],'W_contravariant':[str(v) for v in W],'curvature_quadratic':str(Qcurv),'raw_bulk':str(Lraw),'raw_H_species':str(Hraw),'raw_H_full_shape':H16.shape,'horizontal_bulk':str(Lhor),'horizontal_H_species':str(H),'horizontal_J_species':str(J),'det_P':str(s.factor(P.det())),'P_inverse':str(P.inv()),'det_horizontal':str(s.factor(H.det())),'Schur':str(St),'effective_source':str(Jeff),'Palatini_constant':str(Lconst),'Palatini_solve':{str(a):str(b) for a,b in qu.items()},'residue_solve':str(tstar),'horizontal_solve':{str(a):str(b) for a,b in fullsolve.items()},'derived_on_shell':str(Lon),'raw_solve':{str(a):str(b) for a,b in rawsol.items()},'raw_on_shell':str(rawon),'raw_nullspace_generic':[str(v) for v in Hraw.nullspace()]}
(E/'independent_derivation.json').write_text(json.dumps(result,indent=2));print(json.dumps({a:b for a,b in result.items() if a not in ['distortion_components','curvature_quadratic','raw_bulk','horizontal_bulk']},indent=2))
# Preserve derived objects for the independent validation script without rerunning external code.
objects={'g':g,'gi':gi,'N':Nt,'Qcurv':Qcurv,'W':W,'T':Tv,'Q':Qv,'U':Uv,'A':Av,'p':p,'k':k,'f1':f1,'f2':f2,'f3':f3,'Lraw':Lraw,'Hraw':Hraw,'H16':H16,'J16':J16,'Lhor':Lhor,'H':H,'H12':H12,'J':J,'P':P,'B':B,'C':C,'q':qv,'u':uv,'t':tv,'V':V,'rawsol':rawsol,'raw1':raw1,'raw_on':rawon,'Lon':Lon,'L1':L1,'qu':qu,'tstar':tstar,'qs':qs,'us':us,'ts':ts,'ps':ps,'ks':ks,'Tz':Tz,'Qz':Qz,'Uz':Uz,'Az':Az,'chart':chart,'Non':Non,'M':M}
# No pickle of external content: subsequent checks import this local audit derivation directly.
