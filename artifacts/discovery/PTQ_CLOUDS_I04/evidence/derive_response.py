"""Cycle 1: exact algebra for one minimally coupled scalar shear channel.
Run with the pre-existing palpt environment; writes only beside this script.
Proofs and distributional qualifications are in reports, not inferred from asserts.
"""
from pathlib import Path
import json
import sympy as S

checks = {}
def eq(name, lhs, rhs):
    residual = S.simplify(lhs-rhs)
    checks[name] = {'pass': residual == 0, 'residual': str(residual)}
    if residual != 0:
        raise AssertionError((name, residual))

pi = S.pi
t, a, u = S.symbols('t a u', real=True)
m, s, p = S.symbols('m s p', positive=True)
z, sigma = S.symbols('z sigma')
# No angular-average coefficient is supplied.
angular = S.integrate(S.sin(t)**5, (t,0,pi))*S.integrate(
    S.cos(a)**2*S.sin(a)**2,(a,0,2*pi))/(4*pi)
eq('angular_px2_py2', angular, S.Rational(1,15))
# Lorentz-normalized two-body phase space, radial delta Jacobian:
# dPhi2 = 4pi/(4(2pi)^2) p^2/E^2 * E/(2p).
E = S.sqrt(p*p+m*m)
phase = 4*pi/(4*(2*pi)**2)*p*p/E**2 * E/(2*p)
eq('two_body_phase_space', phase, p/(8*pi*E))
# Identical-state completeness 1/2!, |M_xy|^2 = 4 p_x^2 p_y^2.
W = S.Rational(1,2)*phase*4*p**4*angular
beta = S.sqrt(1-4*m*m/s)
W_s = S.simplify(W.subs(p,S.sqrt(s/4-m*m)))
eq('Wightman_shear', W_s, s*s*beta**5/(960*pi))
rho = s*s*beta**5/(1920*pi*pi)
eq('W_equals_2pi_rho', W_s, 2*pi*rho)
# Generic on-shell stress vertex Ward identity; a_dot denotes p dot p'.
adot = S.symbols('adot')
eq('vertex_Ward_each_coefficient', -(-m*m+adot)+(adot-m*m), 0)
# Exact subtracted dispersion integrand (before any integration).
eq('three_subtractions', 1/(sigma-z)-1/sigma-z/sigma**2-z*z/sigma**3,
   z**3/(sigma**3*(sigma-z)))
sig_u = 4*m*m/(1-u*u)
jac = S.diff(sig_u,u)
eq('threshold_change_of_variable',
   S.simplify(u**5/(sig_u*(sig_u-z))*jac),
   2*u**6/(4*m*m-z*(1-u*u)))
coeffs = []
for n in range(4):
    val = S.integrate(u**6*(1-u*u)**n,(u,0,1))/(960*pi*pi*(4*m*m)**(n+1))
    coeffs.append(S.simplify(val))
eq('low_energy_z3', coeffs[0], 1/(26880*pi*pi*m*m))
eq('low_energy_z4', coeffs[1], 1/(483840*pi*pi*m**4))
# Residue in the u integral above threshold determines the imaginary part.
b = S.symbols('b',positive=True)
residue_im = s**3/(960*pi*pi)*pi*b**6/(2*s*b)
eq('retarded_absorption',residue_im,s*s*b**5/(1920*pi))
# Response sign: i[oscillator q(t),q(0)] = sin(Omega t)/Omega.
O, w, delta = S.symbols('Omega w delta',positive=True,real=True)
osc = (1/(delta-S.I*(w+O))-1/(delta-S.I*(w-O)))/(2*S.I*O)
eq('Kubo_sign_oscillator',osc,1/(O**2-(w+S.I*delta)**2))
# Cross-check the published spin-two convention only after deriving ours.
rho_T = beta*(s/4-m*m)**2/(60*pi*pi)
eq('published_tensor_xy_half',rho,rho_T/2)
# Seagull directly from the same metric-coupled scalar action.
h=S.symbols('h',real=True)
dt,dx,dy,dz,phi=S.symbols('dt dx dy dz phi',real=True)
g=S.Matrix([[-1,0,0,0],[0,1,h,0],[0,h,1,0],[0,0,0,1]])
v=S.Matrix([dt,dx,dy,dz])
L=-S.sqrt(1-h*h)*((v.T*g.inv()*v)[0]+m*m*phi*phi)/2
C=-dt*dt+dx*dx+dy*dy+dz*dz+m*m*phi*phi
eq('linear_metric_vertex',S.diff(L,h).subs(h,0),dx*dy)
eq('explicit_seagull',S.diff(L,h,2).subs(h,0),C/2-dx*dx-dy*dy)
d,B=S.symbols('d B')
AA=-m*m*B/d
eq('dimreg_seagull_coincidence',(d*AA+m*m*B)/2-2*AA,2*m*m*B/d)

out={'epistemic':'Exact algebra; distributional causality is proved in report 01.',
     'sympy':S.__version__,'checks':checks,'low_energy_coefficients_z3_through_z6':[str(v) for v in coeffs]}
Path(__file__).with_name('response_symbolic.json').write_text(json.dumps(out,indent=2)+'\n')
print(f'{len(checks)} exact checks passed')
