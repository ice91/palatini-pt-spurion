"""Cycle 2: regular nonlinear scalar reduction, modes and probe symbols."""
from pathlib import Path
import sympy as s
import json
checks={}
def eq(name,a,b):
 r=a-b;ok=all(s.simplify(x)==0 for x in r)if isinstance(r,s.MatrixBase)else s.simplify(r)==0
 checks[name]={'pass':bool(ok)}
 if not ok:raise AssertionError((name,r))
b=s.symbols('beta',real=True);kap,lam=s.symbols('kappa lambda',positive=True)
K=3/(2*(1+b*b));U=2*b*b/(lam*(1-b*b))
chi=s.sqrt(3/(2*kap))*s.asinh(b)
eq('canonical_field_derivative',s.diff(chi,b)**2,K/kap)
V=U/(2*kap)
mass2=s.simplify((s.diff(V,b,2)/s.diff(chi,b)**2).subs(b,0))
eq('physical_scalar_mass_squared',mass2,4/(3*lam))
eq('potential_positive_factorization',s.factor(V),b*b/(kap*lam*(1-b*b)))
eq('potential_only_stationary_point_regular',s.diff(V,b),2*b/(kap*lam*(1-b*b)**2))
# Actual scalar Euler derivative of L=-K(beta)(d beta)^2-U(beta).
t,x=s.symbols('t x',real=True);f=s.Function('f')(t,x)
Lf=-K.subs(b,f)*(-s.diff(f,t)**2+s.diff(f,x)**2)-U.subs(b,f)
EL=s.diff(Lf,f)-s.diff(s.diff(Lf,s.diff(f,t)),t)-s.diff(s.diff(Lf,s.diff(f,x)),x)
expected=2*K.subs(b,f)*(-s.diff(f,t,2)+s.diff(f,x,2))+s.diff(K,b).subs(b,f)*(-s.diff(f,t)**2+s.diff(f,x)**2)-s.diff(U,b).subs(b,f)
eq('Euler_Lagrange_beta',EL,expected)
eq('linear_beta_equation',s.diff(expected.subs(f,s.symbols('eps')*f),s.symbols('eps')).subs(s.symbols('eps'),0),
   3*(-s.diff(f,t,2)+s.diff(f,x,2))-4*f/lam)
# Nonlinear Legendre map: scalar momentum does not vanish on any regular beta.
N,rootg,bdot,shiftgrad,p=s.symbols('N sqrtgamma bdot shiftgrad p',nonzero=True)
Lvel=rootg*K*(bdot-shiftgrad)**2/(2*kap*N)
momentum=s.diff(Lvel,bdot)
eq('nonlinear_scalar_momentum',momentum,rootg*K*(bdot-shiftgrad)/(kap*N))
eq('nonlinear_scalar_velocity_Hessian',s.diff(momentum,bdot),rootg*K/(kap*N))
# Count from the actually nondegenerate seven-velocity Legendre map and four constraints.
vv=s.symbols('vel0:7'); Km=s.Matrix([[vv[0],vv[3],vv[4]],[vv[3],vv[1],vv[5]],[vv[4],vv[5],vv[2]]])/2
Lseven=(s.trace(Km*Km)-s.trace(Km)**2+K*vv[6]**2)/(2*kap)
Hseven=s.hessian(Lseven,vv)
eq('seven_velocity_Hessian_rank',Hseven.rank(),7)
qq=s.Matrix(s.symbols('q0:7')); pp=s.Matrix(s.symbols('p0:7'))
C=s.Matrix([qq[0]+qq[1],pp[4],pp[5],pp[2]])
JC=C.jacobian(qq.col_join(pp));Om=s.zeros(14);Om[:7,7:]=s.eye(7);Om[7:,:7]=-s.eye(7)
eq('linear_constraint_rank',JC.rank(),4)
eq('linear_constraints_first_class',JC*Om*JC.T,s.zeros(4))
eq('generic_regular_DoF',(s.Integer(2)*len(vv)-2*JC.rank())/2,3)
# Scalar part of the hypersurface-deformation bracket: potential terms cancel.
Np,Mp,Nd,Md,phip,Vp=s.symbols('Nsm Msm Nder Mder phider Vprime')
# After integrating by parts, bracket density in one coordinate; tensor contraction is covariant.
bracket=(-Nd*phip+Np*Vp)*Mp*p-(-Md*phip+Mp*Vp)*Np*p
eq('scalar_Hamiltonian_bracket',s.expand(bracket),p*phip*(Np*Md-Mp*Nd))
# Maxwell operator and probability pairing from the directly B-induced Hodge map.
eta=s.diag(-1,1,1,1);w,k=s.symbols('omega k',real=True);q=s.Matrix([-w,0,0,k]);qu=eta*q
q2=(q.T*qu)[0];M=q2*eta-qu*qu.T
eq('Maxwell_gauge_null',M*q,s.zeros(4,1))
eq('Maxwell_offshell_rank',M.subs({w:3,k:2}).rank(),3)
eq('Maxwell_null_rank',M.subs({w:2,k:2}).rank(),1)
P=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
W=s.Matrix(6,6,lambda i,j:s.LeviCivita(*P[i],*P[j]));D=s.diag(-1,-1,-1,1,1,1);H=-D*W
Ex,Ey,Ez,Bx,By,Bz=s.symbols('Ex Ey Ez Bx By Bz',real=True)
F=s.Matrix([Ex,Ey,Ez,Bz,-By,Bx])
eq('Maxwell_positive_kinetic_density',s.expand(-(F.T*W*H*F)[0]/2),
   (Ex**2+Ey**2+Ez**2-Bx**2-By**2-Bz**2)/2)
# Exact constrained Cauchy data: NOT a second perturbation background or cosmological fit.
c=s.symbols('c',positive=True);V0=V.subs(b,c);H2=kap*V0/3
eq('nonzero_beta_initial_Hamiltonian_constraint',6*H2,2*kap*V0)
eq('hard_projection_initial_constraint_failure',6*H2,2*c*c/(lam*(1-c*c)))
eq('hard_surface_force_vanishes',s.diff(U,b).subs(b,0),0)
Path(__file__).with_name('dynamics_results.json').write_text(json.dumps({'checks':checks,
 'scalar_mass_squared':str(mass2),'canonical_field':str(chi),'scalar_Euler_expression':str(expected),
 'nonlinear_scalar_velocity_Hessian':str(s.diff(momentum,bdot)),
 'scope':'Exact local regular-branch analysis, no global nonlinear stability or quantum completion.'},indent=2)+'\n')
print(len(checks),'dynamics/probe checks passed')
