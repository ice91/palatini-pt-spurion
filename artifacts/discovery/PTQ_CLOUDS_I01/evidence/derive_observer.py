"""Exact identities for the fixed derivative-field parent. No prior model imports."""
from pathlib import Path
import json
import sympy as S

HERE=Path(__file__).resolve().parent
checks={}
def check(name, expression):
    value=S.simplify(S.expand_trig(expression))
    assert value==0, (name,value)
    checks[name]=True

a=S.symbols('a',positive=True)
s,z,E=S.symbols('s z E',real=True)
t,tt=S.symbols('tau tauprime',real=True)
# Null-coordinate differences of the accelerated trajectory; differentiating
# both logarithms avoids an assumed thermal state and the scalar zero mode.
u=lambda x:-S.exp(-a*x)/a
v=lambda x:S.exp(a*x)/a
C=-S.diff(S.log(u(t)-u(tt))+S.log(v(t)-v(tt)),t,tt)/(4*S.pi)
Ca=-a*a/(8*S.pi*S.sinh(a*(t-tt)/2)**2)
check('null_coordinate_derivative_correlator',S.expand_trig((C-Ca).rewrite(S.exp)))
check('inertial_limit',S.limit(Ca.subs({t:s,tt:0}),a,0)+1/(2*S.pi*s*s))
F=-a*a/(8*S.pi*S.sinh(a*z/2)**2)
check('analytic_imaginary_periodicity',F.subs(z,z-2*S.pi*S.I/a)-F)
check('midstrip_kernel',F.subs(z,s-S.I*S.pi/a)-a*a/(8*S.pi*S.cosh(a*s/2)**2))
# Residue at any double pole equals the translated residue at zero.
res=S.residue(S.exp(-S.I*E*z)*F,z,0)
check('double_pole_residue',res-S.I*E/(2*S.pi))
R=E/(S.exp(2*S.pi*E/a)-1)
check('detailed_balance',R/R.subs(E,-E)-S.exp(-2*S.pi*E/a))
check('zero_gap_limit',S.limit(R,E,0)-a/(2*S.pi))
check('state_independent_commutator_spectrum',R.subs(E,-E)-R-E)
check('kms_fluctuation_dissipation',((R.subs(E,-E)+R)-E/S.tanh(S.pi*E/a)).rewrite(S.exp))
# Adversarial counterstate in the SAME free-field theory, not an additional
# observer protocol: a normalized right-moving one-particle wave packet.
ell,k,U=S.symbols('ell k U',positive=True)
check('wave_packet_normalization',S.integrate(4*ell**2*k*S.exp(-2*ell*k),(k,0,S.oo))-1)
f=ell/S.sqrt(S.pi)/(ell+S.I*u(t))
df=S.diff(f,t)
excess=S.simplify(2*df*S.conjugate(df))
target=2*ell**2*S.exp(-2*a*t)/(S.pi*(ell**2+S.exp(-2*a*t)/a**2)**2)
check('nonstationary_one_particle_excess',excess-target)
assert S.simplify(target.subs({a:1,ell:1,t:0})-target.subs({a:1,ell:1,t:S.log(2)}))!=0
checks['wedge_restriction_alone_not_thermal_counterexample']=True

q=S.symbols('q',positive=True)
r=S.symbols('r',positive=True)
n=S.symbols('n',integer=True,nonnegative=True)
# Geometric sums evaluated on 0<q<1. Use r to make convergence domain explicit.
Z=1/(1-r)
mean=S.simplify((r*S.diff(Z,r))/Z)
purity=S.simplify((1-r)**2/(1-r*r))
check('thermal_number',mean-r/(1-r))
check('marginal_purity',purity-(1-r)/(1+r))
cross=(1-q*q)*q*S.diff(1/(1-r),r).subs(r,q*q)
check('cross_wedge_correlation',cross-q/(1-q*q))

x,xp,y=S.symbols('x xp y',real=True)
A=(1+q*q)/(1-q*q)
Cg=2*q/(1-q*q)
check('mehler_precision_determinant',A*A-Cg*Cg-1)
# Actual Gaussian integration by completing the square. The y integral has
# precision 2A, linear coefficient C(x+x').
logrho=-A*(x*x+xp*xp)/2+Cg*Cg*(x+xp)**2/(4*A)
check('position_schur',A-Cg*Cg/A-1/A)
check('diagonal_density_exponent',logrho.subs(xp,x)+x*x/A)
second=S.diff(logrho,x,2)+S.diff(logrho,x)**2
pvar=-S.simplify(second.subs(xp,x).expand().coeff(x,2)*A/2+second.subs({xp:0,x:0}))
check('momentum_variance_from_offdiagonal_kernel',pvar-A/2)
check('thermal_covariance_number',(A-1)/2-q*q/(1-q*q))

# Finite d regulator of the SAME wedge-mode restriction. No truncated CCR
# assumption: exact finite matrices are used throughout.
d=3; q0=S.Rational(1,2)
amps=S.Matrix([q0**i for i in range(d)])
norm=(amps.T*amps)[0]
psi=S.zeros(d*d,1)
for i in range(d): psi[i*d+i]=amps[i]/S.sqrt(norm)
rho=psi*psi.T
Ks=[]
for j in range(d):
    K=S.zeros(d,d*d)
    for i in range(d):K[i,i*d+j]=1
    Ks.append(K)
out=sum((K*rho*K.T for K in Ks),S.zeros(d))
check('finite_marginal',sum(out[i,i]-q0**(2*i)/norm for i in range(d)))
assert out==S.diag(*[q0**(2*i)/norm for i in range(d)])
assert sum((K.T*K for K in Ks),S.zeros(d*d))==S.eye(d*d)
checks['kraus_trace_preservation']=True
J=S.zeros(d*d*d)
for K in Ks:
    vec=S.Matrix([K[o,i] for i in range(d*d) for o in range(d)])
    J+=vec*vec.T
eig=J.eigenvals()
assert eig=={S.Integer(d):d,S.Integer(0):d*d*d-d}
checks['choi_complete_positivity']=True
N=S.diag(*range(d)); B=S.kronecker_product(N,S.eye(d))-S.kronecker_product(S.eye(d),N)
P=S.diag(*[int(j==0) for i in range(d) for j in range(d)])
Q=S.eye(d*d)-P
assert P*P==P and P*B*Q==S.zeros(d*d) and B*psi==S.zeros(d*d,1)
checks['boost_feshbach_cross_block_zero']=True
comm=N*out-out*N
assert comm==S.zeros(d)
checks['restricted_stationarity']=True
K0=Ks[0]; outcome=K0*rho*K0.T
assert S.trace(outcome)==1/norm and outcome/S.trace(outcome)==S.diag(1,0,0)
checks['postselection_not_thermal_partial_trace']=True

result={'all_pass':all(checks.values()),'checks':checks,'count':len(checks),
        'correlator':str(Ca),'rate':str(R),'pole_residue':str(res),
        'nonstationary_counterstate_excess':str(target),
        'gaussian_log_density_kernel':str(logrho),
        'finite_regulator':{'d':d,'q':str(q0),'marginal':str(out),
        'choi_eigenvalues':{str(k):int(v) for k,v in eig.items()},
        'postselection_probability':str(1/norm)},
        'scope':'exact symbolic checks; continuum KMS boundaries and no-go proofs in reports'}
(HERE/'symbolic_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
