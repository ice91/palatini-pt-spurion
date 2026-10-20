"""Cycle 2: separately implemented quadratures, Ward and feedback controls.
No detector data or fitted physical numbers. m=1 fixes units for examples.
"""
from pathlib import Path
import json
import mpmath as mp

mp.mp.dps=45
pi=mp.pi
results={}
def check(name, condition, **data):
    results[name]={'pass':bool(condition), **{k:str(v) for k,v in data.items()}}
    if not condition: raise AssertionError((name,data))
def rho(s,m=1):
    return mp.mpf(0) if s<=4*m*m else s*s*(1-4*m*m/s)**mp.mpf('2.5')/(1920*pi*pi)
def spectral(z,m=1):
    return z**3*mp.quad(lambda s:rho(s,m)/(s**3*(s-z)),[4*m*m,8*m*m,mp.inf])
def compact(z,m=1):
    return z**3/(960*pi*pi)*mp.quad(lambda u:u**6/(4*m*m-z*(1-u*u)),[0,1])
# Independently integrate the Wick contraction on a COM shell, not |M| supplied.
for energy in [mp.mpf(3),mp.mpf(5)]:
    p=mp.sqrt(energy**2/4-1); E=energy/2
    angular=mp.quad(lambda th:mp.sin(th)**5,[0,pi])*mp.quad(
        lambda ph:mp.cos(ph)**2*mp.sin(ph)**2,[0,pi,2*pi])
    # Wick factor 2, delta(Etotal-2E) gives E/(2p).
    W=2/(4*(2*pi)**2)*p**6/E**2*(E/(2*p))*angular
    target=2*pi*rho(energy**2)
    check('canonical_Wick_shell_'+str(energy),abs(W/target-1)<mp.mpf('1e-40'),W=W,target=target)
for z in [-3,-mp.mpf('.4'),mp.mpf('.7'),mp.mpc('1.2','.6'),mp.mpc('5','.9')]:
    x,y=spectral(z),compact(z)
    check('two_integral_representations_'+str(z),abs(x-y)<mp.mpf('1e-40'),spectral=x,compact=y)
# Full tensor vertex conservation after a boost in z, independent of symbolic Ward.
E=mp.mpf('2'); p=mp.sqrt(E*E-1)
pv=[E,p/mp.sqrt(3),p/mp.sqrt(3),p/mp.sqrt(3)]
rv=[E,-pv[1],-pv[2],-pv[3]]
v=mp.mpf('.37'); gamma=1/mp.sqrt(1-v*v)
def boost(x): return [gamma*(x[0]+v*x[3]),x[1],x[2],gamma*(x[3]+v*x[0])]
pv,rv=boost(pv),boost(rv)
eta=[-1,1,1,1]; pc=[eta[i]*pv[i] for i in range(4)]; rc=[eta[i]*rv[i] for i in range(4)]
dot=sum(pc[i]*rv[i] for i in range(4)); q=[pv[i]+rv[i] for i in range(4)]
M=[[-pc[i]*rc[j]-pc[j]*rc[i]+(eta[i]*(dot-1) if i==j else 0)
    for j in range(4)] for i in range(4)]
ward=max(abs(sum(q[i]*M[i][j] for i in range(4))) for j in range(4))
check('boosted_stress_Ward',ward<mp.mpf('1e-40'),max_residual=ward)
# Independent near-cut test via complex integral, with a split around its pole.
absorption=[]
s=mp.mpf(9); u0=mp.sqrt(1-4/s)
for eps in [mp.mpf('.01'),mp.mpf('.001'),mp.mpf('.0001')]:
    z=mp.mpc(s,eps)
    val=z**3/(960*pi*pi)*mp.quad(lambda u:u**6/(4-z*(1-u*u)),
                                    [0,u0-mp.mpf('.03'),u0,u0+mp.mpf('.03'),1])
    absorption.append(abs(val.imag/(pi*rho(s))-1))
check('retarded_cut_limit',absorption[-1]<mp.mpf('.0002') and
      absorption[2]<absorption[1]<absorption[0],relative_errors=absorption)
check('negative_frequency_conjugation',abs(compact(mp.mpc(5,1)).conjugate()-compact(mp.mpc(5,-1)))<mp.mpf('1e-40'))
check('below_threshold_no_absorption',compact(mp.mpf(3)).imag==0,
      real_response=compact(mp.mpf(3)))
# Regular branch: analytic bound in report 02; sample disk checks corroborate, not prove.
Z=mp.mpf(1); b=mp.mpf('.2'); L2=mp.mpf('.5'); a=1/(26880*pi*pi)
eta_bound=abs(b)*L2/Z+a*L2**2/(Z*(1-L2/4))
worst=mp.mpf(0); error=mp.mpf(0)
for j in range(24):
    z=L2*mp.exp(2j*pi*j/24); correction=(b*z*z+compact(z))/(Z*z)
    worst=max(worst,abs(correction))
    error=max(error,abs(1/(1+correction)-(1-correction)))
check('regular_branch_bound',worst<=eta_bound<1,bound=eta_bound,observed=worst)
check('order_reduction_error',error<=eta_bound**2/(1-eta_bound),
      observed=error,bound=eta_bound**2/(1-eta_bound))
# Adversarial resummed higher-derivative metric operator, same matter kernel.
# Such a pole is outside the perturbative gravity branch, even if below m.
b=mp.mpf(4)
def D(z):return Z*z+b*z*z+compact(z)
root=mp.findroot(D,(-mp.mpf('.26'),-mp.mpf('.24')))
check('resummed_runaway_counterexample',root<0 and abs(D(root))<mp.mpf('1e-40'),
      z_root=root,upper_half_omega=mp.sqrt(-root),residual=D(root),
      local_expansion_ratio=abs(b*root/Z))
check('runaway_IVT_bound',b*b>4*a*Z and D(-2*Z/b)>0,
      D_test=D(-2*Z/b),D_near_zero=D(-mp.mpf('.001')))
# Same rho/chi, two positive G choices: distinct source-to-metric response.
z=-mp.mpf('.1'); h1=-1/(1*z+compact(z)); h2=-1/(2*z+compact(z))
check('matter_does_not_fix_G',h1!=h2,response_Z1=h1,response_Z2=h2,
      common_chi=compact(z))
out={'python_precision_decimal':mp.mp.dps,'results':results,
     'scope':'Numerical checks of analytic derivations, not experiments or a gravity proof.'}
Path(__file__).with_name('independent_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(f'{len(results)} independent check cases passed')
