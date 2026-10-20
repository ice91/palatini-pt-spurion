"""Scoped checks of supplied historical equations, not a new physical model.
No repository code, historical notebooks, or downloaded code is executed.
"""
import sympy as s
import json,sys
from pathlib import Path
checks=[]
def check(name,condition,result):
 assert condition,name
 checks.append({'name':name,'result':str(result),'status':'PASS'})
T,v,c,m2=s.symbols('T v c m2');eq=m2*(T-v)+c*T
sol=s.solve(eq,T)[0]
check('finite_written_equation_solution',s.simplify(sol-m2*v/(m2+c))==0,sol)
check('claimed_exact_T_equals_v_residual',s.simplify(eq.subs(T,v))==c*v,eq.subs(T,v))
check('finite_invariant_trace',s.simplify(sol-v+c*v/(m2+c))==0,s.simplify(sol-v))
check('hard_penalty_limit',s.limit(sol,m2,s.oo)==v,s.limit(sol,m2,s.oo))
check('positive_mass_alone_not_invertibility',s.simplify((m2+c).subs(c,-m2))==0,'m2>0,c=-m2: denominator zero')
eta,Sigma=s.symbols('eta Sigma');norm=6*eta**2*Sigma
check('quarter_IT_contraction',-norm/4==-s.Rational(3,2)*eta**2*Sigma,-norm/4)
check('main_Symmetry_IT_contraction',-norm==-6*eta**2*Sigma,-norm)
# Literal printed sign placement, not an automatically corrected projector.
x=s.symbols('x');n_updn_timelike=s.diag(-1,0,0,0);n_updn_spacelike=s.diag(0,1,0,0)
for label,sgn,N in [('timelike',-1,n_updn_timelike),('spacelike',1,n_updn_spacelike)]:
 M=sgn*(N-s.eye(4)/4);jac=s.sqrt((s.eye(4)+s.Rational(2,3)*x*M).det());series=s.series(jac-1,x,0,3).removeO()
 exp=2*x/3+x*x/36 if sgn==-1 else -x*x/12
 check('literal_Tcal_'+label,s.simplify(series-exp)==0,{'trace':str(s.trace(M)),'trace_square':str(s.trace(M*M)),'jacobian_minus1':str(series)})
 check('actual_CM_'+label,s.series(s.sqrt(1+s.Rational(2,3)*sgn*x)-1,x,0,3).removeO()==sgn*x/3-x*x/18,sgn*x/3-x*x/18)
R=s.Rational
before=s.Matrix([[R(3,4),-R(1,2)],[R(1,5),R(7,20)]]);after=s.Matrix([[R(3,4),-R(1,2)],[R(1,5),-R(2,15)]])
check('historical_matrix_before_change',before.det()==R(29,80) and before.rank()==2,before.det())
check('historical_matrix_after_intended_rational',after.det()==0 and after.nullspace()==[s.Matrix([R(2,3),1])],after.nullspace())
# Test precisely the claimed two constraints in v2 Lemma11 on its independent variables.
Lam,pT,pLam,a=s.symbols('Lambda pT pLambda a');coords=[T,Lam];mom=[pT,pLam]
def PB(f,h):return sum(s.diff(f,q)*s.diff(h,p)-s.diff(f,p)*s.diff(h,q) for q,p in zip(coords,mom))
chi=T-a*v; C=s.Matrix([[PB(f,h) for h in [pLam,chi]] for f in [pLam,chi]])
check('v2_Lemma11_displayed_pair_not_second_class',C==s.zeros(2),C)
# No field-dependent NY rescue: d(eps Q)=d eps wedge Q+eps dQ.
z=s.symbols('z'); e=s.Function('epsilon')(z); Q=s.Function('Q')(z)
check('nonconstant_weighted_exact_form_product_rule',s.simplify(s.diff(e*Q,z)-s.diff(e,z)*Q-e*s.diff(Q,z))==0 and s.diff(e,z)*Q!=0,'eps*dQ = d(eps*Q) - d eps wedge Q; second term requires evaluation')
# Sign of the printed +b spatial-four-derivative term with the displayed EH kinetic sign.
w,k,b,L=s.symbols('omega k b Lambda');kernel=w*w-k*k+b*k**4/L**2
check('old_NLO_plus_spatial_term_dispersion_sign',s.solve(kernel,w*w)[0]==k*k-b*k**4/L**2,s.solve(kernel,w*w)[0])
print(json.dumps({'python':sys.version,'sympy':s.__version__,'checks':checks,'passed':len(checks)},indent=2,ensure_ascii=False))
