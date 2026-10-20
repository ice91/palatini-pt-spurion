"""I02: modular/relative-energy calculation for I01's fixed current QFT.
No project imports, no fictitious continuum density matrix.
"""
from pathlib import Path
import json
import sympy as S
import mpmath as mp
HERE=Path(__file__).resolve().parent
checks={};expressions={}
def zero(name,value):
    terms=list(value) if isinstance(value,S.MatrixBase) else [value]
    assert all(S.simplify(S.expand_trig(v))==0 for v in terms),(name,value)
    checks[name]=True

t,x,eta,s,lam=S.symbols('t x eta s lambda',real=True)
metric=S.diag(1,-1)
L=S.Matrix([[S.cosh(eta),S.sinh(eta)],[S.sinh(eta),S.cosh(eta)]])
zero('boost_is_lorentz',L.T*metric*L-metric)
zero('modular_boost_generator',L.subs(eta,-2*S.pi*s).diff(s).subs(s,0)+2*S.pi*S.Matrix([[0,1],[1,0]]))
phi=S.Function('phi')(t,x)
action=(S.diff(phi,t)**2-S.diff(phi,x)**2)/2
EL=S.diff(S.diff(action,S.diff(phi,t)),t)+S.diff(S.diff(action,S.diff(phi,x)),x)
zero('parent_EL_is_wave',EL-S.diff(phi,t,2)+S.diff(phi,x,2))
f=S.Function('f')
cl=lam*(f(x-t)+f(x+t))/2
zero('coherent_mean_solves_wave',S.diff(cl,t,2)-S.diff(cl,x,2))
energy=(S.diff(cl,t)**2+S.diff(cl,x)**2)/2
flux=-S.diff(cl,t)*S.diff(cl,x)
zero('stress_energy_conservation',S.diff(energy,t)+S.diff(flux,x))
zero('initial_energy_from_shift',energy.subs(t,0)-lam**2*S.diff(f(x),x)**2/2)
zero('initial_momentum_zero',S.diff(cl,t).subs(t,0))
# The unitary shift U†phi U=phi+lambda f and U†pi U=pi gives these
# actual quadratic shifts, including linear fluctuation terms.
p,h,fp=S.symbols('pi h fprime',real=True)
delta=((p*p+(h+lam*fp)**2)-(p*p+h*h))/2
zero('normal_ordered_quadratic_shift',delta-lam*h*fp-lam**2*fp**2/2)

# Finite I01 boost mode: q=1/2,d=3. It is a regulator illustration of
# differentiating states, not a replacement for the continuum wedge algebra.
eps=S.symbols('epsilon',real=True)
ps=[S.Rational(16,21),S.Rational(4,21),S.Rational(1,21)]
sigma=S.diag(*ps)
X=S.Matrix([[-1,1,0],[1,1,0],[0,0,0]])
rho=sigma+eps*X
K=S.diag(*[-S.log(p) for p in ps])
zero('perturbation_normalized',S.trace(rho)-1)
zero('fixed_reference_deltaK',S.trace((rho-sigma)*K)-eps*S.log(4))
disc=S.sqrt((S.Rational(2,7)-eps)**2+eps**2)
ev=[S.Rational(10,21)+disc,S.Rational(10,21)-disc,ps[2]]
z=S.symbols('z')
zero('noncommuting_eigenvalues',rho.charpoly(z).as_expr()-S.prod(z-v for v in ev))
entropy=-sum(v*S.log(v) for v in ev)
S0=-sum(p*S.log(p) for p in ps)
D=eps*S.log(4)-(entropy-S0)
zero('nontrivial_first_law',S.diff(entropy,eps).subs(eps,0)-S.log(4))
zero('relative_entropy_stationary',S.diff(D,eps).subs(eps,0))
hessian=S.Rational(105,16)+S.Rational(7,2)*S.log(4)
zero('relative_entropy_positive_hessian',S.diff(D,eps,2).subs(eps,0)-hessian)
# Independent Frechet/log resolvent expression, evaluated entrywise.
frechet=sum(X[i,i]**2/ps[i] for i in range(3))
frechet+=sum(X[i,j]**2*(S.log(ps[i])-S.log(ps[j]))/(ps[i]-ps[j]) for i in range(3) for j in range(3) if i!=j)
zero('hessian_equals_log_frechet_metric',frechet-hessian)
diagprob=[ps[0]-eps,ps[1]+eps,ps[2]]
Ddiag=sum(q*S.log(q/p) for q,p in zip(diagprob,ps))
zero('diagonal_first_law',S.diff(-sum(q*S.log(q) for q in diagprob),eps).subs(eps,0)-S.log(4))
zero('diagonal_relative_entropy_quadratic',S.diff(Ddiag,eps,2).subs(eps,0)-S.Rational(105,16))

# Actual transformed coherent Cauchy data under modular conjugation of a
# Weyl unitary: inverse spacetime boost, eta=+2*pi*s on its classical mean.
ts=t*S.cosh(2*S.pi*s)+x*S.sinh(2*S.pi*s)
xs=x*S.cosh(2*S.pi*s)+t*S.sinh(2*S.pi*s)
clboost=cl.subs({t:ts,x:xs},simultaneous=True)
zero('coherent_modular_field_generator_at_t0',S.diff(clboost,s).subs({s:0,t:0}))
zero('coherent_modular_momentum_generator',S.diff(clboost,s,t).subs({s:0,t:0})-2*S.pi*lam*S.diff(x*S.diff(f(x),x),x))

# SAME I01 one-particle counterstate used by the subordinate detector check:
# f_k=2 ell sqrt(k) exp(-ell k), F(U)=ell/[sqrt(pi)*(ell+iU)].
# Its modular-energy FORM is computable even though its Araki relative
# entropy is not computed in this sprint.
ellp,xp=S.symbols('ell_packet x_packet',positive=True)
one_stress=2*ellp**2/(S.pi*(ellp**2+xp**2)**2)
zero('one_particle_total_energy',2*S.integrate(one_stress,(xp,0,S.oo))-1/ellp)
zero('one_particle_right_modular_energy',2*S.pi*S.integrate(xp*one_stress,(xp,0,S.oo))-2)

# A C-infinity packet b(y)=exp[-1/(1-y²)] inside (-1,1). Exact statements
# use parity and variable substitution; integration below supplies numbers.
mp.mp.dps=50
def bp(y):
    if abs(y)>=1:return mp.mpf(0)
    return -2*y*mp.exp(-1/(1-y*y))/(1-y*y)**2
I=mp.quad(lambda y:bp(y)**2,[-1,0,1])
odd=mp.quad(lambda y:y*bp(y)**2,[-1,0,1])
assert I>0 and abs(odd)<mp.mpf('1e-45')
checks['compact_packet_integrals_finite_and_even']=True
amplitude=mp.mpf(1)/5;width=mp.mpf(1);center=mp.mpf(3)
Ecl=amplitude**2*I/(2*width)
Dcoh=2*mp.pi*center*Ecl
expressions.update({'boost_matrix':str(L),'modular_flow_parameter':'eta=-2*pi*s',
  'coherent_solution':str(cl),'coherent_energy_density':str(energy.subs(t,0)),
  'rho_epsilon':str(rho),'rho_eigenvalues':[str(v) for v in ev],
  'deltaK_regulator':str(eps*S.log(4)),
  'first_law_slope':str(S.log(4)),
  'relative_entropy_hessian':str(hessian),
  'relative_entropy_formula':str(D),
  'positivity_domain':'|epsilon|<1/20 is a sufficient faithful-state neighborhood',
  'coherent_continuum':'D_R(omega_lambda||omega_0)=2*pi*d*E_cl=pi*lambda^2*d*I_b/L',
  'opposite_means_same_stress':'D_R(omega_lambda||omega_minus_lambda)=4*D_R(omega_lambda||omega_0)',
  'translated_wedge':'D_b=2*pi*(d-b)*E_cl for b<d-L',
  'one_particle_stress_at_t0':str(one_stress),
  'one_particle_energy':'1/ell',
  'one_particle_right_deltaK':'2 (finite vacuum-subtracted form); Araki D not computed',
  'packet_illustration':{'I_b':str(I),'odd_moment':str(odd),'lambda':'1/5','L':'1','d':'3',
                         'Minkowski_energy':str(Ecl),'relative_entropy_nats':str(Dcoh)}})
result={'all_pass':all(checks.values()),'count':len(checks),'checks':checks,'expressions':expressions,
        'scope':'formal exact algebra plus a declared compact-packet quadrature; continuum positivity/domain proof is in report01'}
(HERE/'modular_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'all_pass':result['all_pass'],'count':result['count'],'expressions':expressions},indent=2))
