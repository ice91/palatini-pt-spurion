"""Cycle 2 geometry check from the metric, not configured ADM coefficients.
Homogeneous h_xy(t) fixes tensor kinetic and Ricci-squared normalizations;
Lorentz covariance supplies their t,z extension, proved in report 02.
"""
from pathlib import Path
import json
import sympy as S
t, lam=S.symbols('t lam',real=True)
f=S.Function('f')(t)
g=S.Matrix([[-1,0,0,0],[0,1,lam*f,0],[0,lam*f,1,0],[0,0,0,1]])
ginv=g.inv()
def d(expr,i): return S.diff(expr,t) if i==0 else S.S.Zero
Gamma=[[[S.simplify(sum(ginv[a,b]*(d(g[b,c],j)+d(g[b,j],c)-d(g[j,c],b))
                         for b in range(4))/2)
         for c in range(4)] for j in range(4)] for a in range(4)]
Ric=S.zeros(4)
for i in range(4):
 for j in range(4):
  Ric[i,j]=S.simplify(sum(d(Gamma[a][i][j],a)-d(Gamma[a][i][a],j)+
      sum(Gamma[a][a][b]*Gamma[b][i][j]-Gamma[a][j][b]*Gamma[b][i][a]
          for b in range(4)) for a in range(4)))
R=S.simplify(sum(ginv[i,j]*Ric[i,j] for i in range(4) for j in range(4)))
R2=S.expand(S.series(S.sqrt(1-lam**2*f**2)*R,lam,0,3).removeO()).coeff(lam,2)
Ric1=Ric.applyfunc(lambda x:S.simplify(S.diff(x,lam).subs(lam,0)))
eta=S.diag(-1,1,1,1)
Ricnorm=S.simplify(S.trace(eta*Ric1*eta*Ric1))
checks={}
def eq(name,x,y):
 r=S.simplify(x-y); checks[name]={'pass':r==0,'residual':str(r)}
 if r!=0: raise AssertionError((name,r))
eq('EH_raw_quadratic',R2,-S.Rational(3,2)*S.diff(f,t)**2-2*f*S.diff(f,t,2))
eq('EH_after_IBP',R2+2*S.diff(f*S.diff(f,t),t),S.diff(f,t)**2/2)
eq('linear_scalar_curvature_zero',S.diff(R,lam).subs(lam,0),0)
eq('Ricci_squared_quadratic',Ricnorm,S.diff(f,t,2)**2/2)
Z,b=S.symbols('Z b',real=True)
L=Z*S.diff(f,t)**2/2+b*S.diff(f,t,2)**2/2
EL=S.diff(L,f)-S.diff(S.diff(L,S.diff(f,t)),t)+S.diff(S.diff(L,S.diff(f,t,2)),t,2)
eq('metric_variation',EL,-Z*S.diff(f,t,2)+b*S.diff(f,t,4))
z,X=S.symbols('z X')
eq('order_reduced_inverse_identity',1/(1+X)-(1-X),X*X/(1+X))
eq('I03_local_EH_normalization',S.Symbol('J')/(12*S.pi)/(16*S.pi),S.Symbol('J')/(192*S.pi**2))
out={'sympy':S.__version__,'checks':checks,'sqrt_g_R_order_h2':str(R2),
     'Ricci_linear':str(Ric1),'EL':str(EL),
     'scope':'Exact differential-geometry algebra plus compact-source IBP. Not a metric-dynamics postulate.'}
Path(__file__).with_name('feedback_symbolic.json').write_text(json.dumps(out,indent=2)+'\n')
print(f'{len(checks)} geometry/feedback checks passed')
