"""Calculation A: scalar cone spectrum -> replica area entropy.
No assumed Newton coupling or inserted target area coefficient.
"""
from pathlib import Path
import json
import sympy as S
HERE=Path(__file__).resolve().parent
checks={};out={}
def zero(name,expr):
    assert S.simplify(expr)==0,(name,expr)
    checks[name]=True
n,u,s,p,A,J=S.symbols('n u s p A J',positive=True)
# Cone angle 2*pi*n. Angular orders |ell|/n; Laplace transformed radial
# I_nu integral is exp(-nu*u)/sinh(u), p=cosh(u)>1.
angular=1+2/(S.exp(u/n)-1)
zero('geometric_angular_sum',angular-(S.coth(u/(2*n))).rewrite(S.exp))
Tn=S.coth(u/(2*n))/(2*S.sinh(u))
T1=Tn.subs(n,1)
zero('flat_radial_Abel_trace',(T1-1/(2*(S.cosh(u)-1))).rewrite(S.exp))
# Remove the same radially regulated extensive term BEFORE u -> 0.
defect=S.simplify(Tn-n*T1)
C=S.simplify(S.limit(defect,u,0,dir='+'))
zero('computed_conical_defect',C-(1/n-n)/12)
zero('flat_cone_defect_zero',C.subs(n,1))
transverse=S.integrate(2*S.pi*p*S.exp(-s*p*p)/(2*S.pi)**2,(p,0,S.oo))
zero('transverse_heat_trace_per_area',transverse-1/(4*S.pi*s))
Gamma_defect=-A*C*J/(8*S.pi)
entropy=S.simplify((n*S.diff(Gamma_defect,n)-Gamma_defect).subs(n,1))
renyi=S.factor(Gamma_defect/(n-1))
zero('entropy_from_replica_operation',entropy-A*J/(48*S.pi))
zero('renyi_from_normalized_trace',renyi-A*J*(1+1/n)/(96*S.pi))
zero('renyi_limit',S.limit(renyi,n,1)-entropy)
zero('renyi_monotone_derivative',S.diff(renyi,n)+A*J/(96*S.pi*n*n))
# Exact orbital images provide a separate finite-angle check.
image={}
for q in [2,3,4,6]:
    val=S.simplify(sum(1/S.sin(S.pi*k/q)**2 for k in range(1,q))/(4*q))
    zero('orbifold_image_q%d'%q,val-C.subs(n,S.Rational(1,q)))
    image[str(q)]=str(val)
# A finite-angle failure of an EH-only extrapolation is retained.
eh_linear=(1-n)/6
assert S.simplify(C.subs(n,2)-eh_linear.subs(n,2))!=0
checks['finite_replica_angle_not_EH_linearization']=True
out.update({'conical_trace_defect':str(C),'radially_regulated_trace':str(Tn),
            'transverse_trace_density':str(transverse),'Gamma_defect':str(Gamma_defect),
            'entropy':str(entropy),'renyi':str(renyi),'orbifold_images':image,
            'n_equals_2_exact_defect':str(C.subs(n,2)),
            'n_equals_2_EH_linear_defect':str(eh_linear.subs(n,2)),
            'J_definition':'integral_{epsilon^2}^infinity ds exp(-m^2*s)/s^2',
            'assumptions':'minimal real scalar; Friedrichs cone Laplacian; bulk-subtracted infinite cone; plane per-area limit'})
result={'all_pass':all(checks.values()),'count':len(checks),'checks':checks,'expressions':out}
(HERE/'area_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
