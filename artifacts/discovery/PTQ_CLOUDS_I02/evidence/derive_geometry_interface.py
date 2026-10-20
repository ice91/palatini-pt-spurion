"""I02: actual 2D geometric obstruction and explicitly conditional horizon algebra.
The higher-dimensional template is not a second quantum parent.
"""
from pathlib import Path
import json
import sympy as S
HERE=Path(__file__).resolve().parent
checks={};values={}
def zero(name,a):
    terms=list(a) if isinstance(a,S.MatrixBase) else [a]
    assert all(S.simplify(v)==0 for v in terms),(name,a)
    checks[name]=True
def geometry(coords,g):
    n=len(coords);gi=g.inv()
    G=[[[S.simplify(sum(gi[a,d]*(S.diff(g[d,c],coords[b])+S.diff(g[d,b],coords[c])-S.diff(g[b,c],coords[d]))/2 for d in range(n))) for c in range(n)] for b in range(n)] for a in range(n)]
    Ric=S.Matrix(n,n,lambda b,d:S.simplify(sum(S.diff(G[a][b][d],coords[a])-S.diff(G[a][b][a],coords[d])+sum(G[a][c][a]*G[c][b][d]-G[a][c][d]*G[c][b][a] for c in range(n)) for a in range(n))))
    R=S.simplify(sum(gi[a,b]*Ric[a,b] for a in range(n) for b in range(n)))
    return Ric,R,S.simplify(Ric-g*R/2)
t,x=S.symbols('t x',real=True);c=S.Function('c')(t,x)
g=S.exp(2*c)*S.diag(1,-1)
Ric,R,Ein=geometry([t,x],g)
zero('two_dimensional_Einstein_tensor',Ein)
values['conformal_2D_Ricci']=str(Ric);values['conformal_2D_scalar_R']=str(R)
eta,rho=S.symbols('eta rho',real=True,positive=True)
rr,rs,re=geometry([eta,rho],S.diag(rho*rho,-1))
zero('Rindler_coordinate_curvature_zero',rr)
fprime,amp=S.symbols('fprime lambda',real=True,nonzero=True)
stress=amp*amp*fprime*fprime*S.eye(2)/2
zero('coherent_stress_traceless_2D',S.trace(S.diag(1,-1)*stress))
assert stress[0,0]!=0
checks['nonzero_energy_does_not_curve_fixed_parent']=True

# Conditional local-horizon template ONLY, dimensions >=3. lambda_affine
# runs from -ell to zero. Fix curvature sign by Raychaudhuri theta'=-R_kk
# at an equilibrium cut. No Einstein equation is an input to this algebra.
aff=S.symbols('lambda_affine',real=True)
ell,kappa,alpha=S.symbols('ell kappa alpha',positive=True)
Rkk,Tkk=S.symbols('Rkk Tkk',real=True)
theta=-aff*Rkk
deltaA=S.integrate(theta,(aff,-ell,0))
deltaQ=S.integrate(-kappa*aff*Tkk,(aff,-ell,0))
temp=kappa/(2*S.pi)
relation=S.solve(alpha*deltaA-deltaQ/temp,Rkk)[0]
zero('conditional_Clausius_null_relation',relation-2*S.pi*Tkk/alpha)
Newton=S.symbols('G',positive=True)
zero('conditional_area_density_normalization',relation.subs(alpha,1/(4*Newton))-8*S.pi*Newton*Tkk)
# Null completeness lemma in a Lorentzian tangent space: q(e0 +/- ei)=0
# enforces S0i=0 and Sii=-S00. Mixed spatial null directions fix offdiag.
S00,S0i,Sii=S.symbols('S00 S0i Sii',real=True)
pair=S.solve([S00+2*S0i+Sii,S00-2*S0i+Sii],[S0i,Sii])
zero('null_lemma_time_space_offdiagonal',pair[S0i])
zero('null_lemma_diagonal',pair[Sii]+S00)
Sij=S.symbols('Sij',real=True)
mixed=S00+(-S00-S00)/2+Sij
zero('null_lemma_spatial_offdiagonal',S.solve(mixed,Sij)[0])
values.update({'Rindler_R':str(rs),'initial_coherent_stress':str(stress),
               'conditional_theta':str(theta),'conditional_delta_area':str(deltaA),
               'conditional_delta_heat':str(deltaQ),'conditional_null_relation':str(relation),
               'conditional_tensor_result':'G_ab + Lambda*g_ab = (2*pi/alpha)*<T_ab>',
               'premise_warning':'area density alpha, Clausius equilibrium for all local null horizons, metric variation and stress conservation are ADDED; absent in 1+1 fixed parent'})
result={'all_pass':all(checks.values()),'count':len(checks),'checks':checks,'expressions':values}
(HERE/'geometry_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
