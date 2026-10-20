"""Calculation B: smooth scalar heat kernel -> induced local curvature term.
The metric remains a nondynamical source. No metric equation is imposed.
"""
from pathlib import Path
import json
import sympy as S
HERE=Path(__file__).resolve().parent
checks={};out={}
def zero(name,expr):
    assert S.simplify(expr)==0,(name,expr)
    checks[name]=True
ys=S.symbols('y0:4',real=True)
Ric=S.Matrix(4,4,lambda i,j:S.Symbol('R%d%d'%(min(i,j),max(i,j)),real=True))
# Van Vleck square root in Riemann normal coordinates, to required order.
V=1+sum(Ric[i,j]*ys[i]*ys[j] for i in range(4) for j in range(4))/12
a1=S.simplify(sum(S.diff(V,y,2) for y in ys))
zero('minimal_scalar_a1_from_transport',a1-S.trace(Ric)/6)
M,eps=S.symbols('m_squared epsilon',positive=True)
z=M*eps**2
J=S.exp(-z)/eps**2-M*S.expint(1,z)
zero('proper_time_integral_lower_derivative',S.diff(J,eps)+2*S.exp(-z)/eps**3)
zero('proper_time_integral_mass_derivative',S.diff(J,M)+S.expint(1,z))
zero('massless_integral_limit',S.limit(J,M,0,dir='+')-1/eps**2)
a1_factor=S.simplify(S.diff(a1,Ric[0,0]))
cR=-S.Rational(1,2)/(4*S.pi)**2*a1_factor*J
delta_inverseG=S.simplify(-16*S.pi*cR)
sA=J/(48*S.pi)
zero('induced_EH_coefficient',cR+J/(192*S.pi**2))
zero('induced_inverse_Newton',delta_inverseG-J/(12*S.pi))
zero('entropy_curvature_matching',sA+4*S.pi*cR)
zero('entropy_inverseG_matching',sA-delta_inverseG/4)
# Independent smooth-cone geometric integral, rather than putting 4*pi into
# the action by hand. ds²=dr²+F(r)²dtheta², theta in [0,2*pi].
n,r,cap=S.symbols('n r cap_width',positive=True)
F=n*r+(1-n)*cap*S.tanh(r/cap)
Fp=S.diff(F,r)
zero('smooth_cap_axis_derivative',Fp.subs(r,0)-1)
zero('smooth_cap_asymptotic_derivative',S.limit(Fp,r,S.oo)-n)
integratedR=S.simplify(-4*S.pi*(S.limit(Fp,r,S.oo)-Fp.subs(r,0)))
zero('smooth_cap_integrated_curvature',integratedR-4*S.pi*(1-n))
Sgeom=S.simplify((n*S.diff(cR*integratedR,n)-cR*integratedR).subs(n,1))
zero('curvature_term_replica_entropy',Sgeom-sA)
# Quartic source-volume term: cannot simply discard it if the metric is
# subsequently postulated to be dynamical.
J3=S.exp(-z)/(2*eps**4)-M*J/2
zero('volume_integral_lower_derivative',S.diff(J3,eps)+2*S.exp(-z)/eps**5)
c0=-J3/(32*S.pi**2)
zero('volume_massless_limit',S.limit(c0,M,0,dir='+')+1/(64*S.pi**2*eps**4))
# Common-profile countercheck, NOT a new field model. The smooth regulator
# w(t)=1-exp(-t^3) regulates both curvature and volume integrals at t=0.
t=S.symbols('t',positive=True)
smooth_leading=S.integrate(3*t*S.exp(-t**3),(t,0,S.oo))
zero('smooth_profile_massless_leading',smooth_leading-S.gamma(S.Rational(2,3)))
N,bare,ren=S.symbols('N bare_inverseG ren_inverseG',real=True)
zero('species_common_matching',N*sA-N*delta_inverseG/4)
bare_needed=ren-delta_inverseG
zero('arbitrary_finite_Newton_counterterm',bare_needed+delta_inverseG-ren)
zero('area_level_renormalization',bare_needed/4+sA-ren/4)
out.update({'a1_from_transport':str(a1),'J_massive':str(J),
 'J_small_cutoff':str(S.series(J,eps,0,3)),
 'c_R_Euclidean':str(cR),'delta_inverseG':str(delta_inverseG),'entropy_per_area':str(sA),
 'smooth_cap_metric_F':str(F),'integrated_cone_curvature':str(integratedR),
 'replica_entropy_from_curvature_per_area':str(Sgeom),'c_0_volume_Euclidean':str(c0),
 'smooth_regulator_leading_J':str(smooth_leading/eps**2),
 'bare_inverseG_for_any_renormalized_value':str(bare_needed),
 'matching_scope':'same scalar, same proper-time profile, minimal coupling, planar surface, n->1',
 'no_metric_equation':'Gamma is a source functional. Stationarity/integration over g is NOT derived.'})
result={'all_pass':all(checks.values()),'count':len(checks),'checks':checks,'expressions':out}
(HERE/'curvature_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
