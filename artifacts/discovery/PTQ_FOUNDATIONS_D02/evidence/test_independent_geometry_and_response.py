"""Independent D02 calculations, no import of derivation scripts or result JSON."""
from pathlib import Path
from itertools import combinations
import unittest,json
import sympy as s

class IndependentGeometry(unittest.TestCase):
 def test_trace_complete_square_and_24_stationary_variations(self):
  M2,beta=s.symbols('M2 beta',nonzero=True,real=True)
  t=s.Matrix(s.symbols('t0:4',real=True));v=s.Matrix(s.symbols('v0:4',real=True));eta=s.diag(-1,1,1,1)
  Lt=-M2*(t.T*eta*t)[0]/3+beta*(t.T*eta*v)[0]
  solve=s.solve([s.diff(Lt,ti) for ti in t],list(t))
  ts=s.Matrix([solve[ti] for ti in t])
  # Build connection matrices from that analytic trace solve, then vary every compatible direction.
  W=[]
  for mu in range(4):
   W.append(s.Matrix(4,4,lambda a,b:((eta[b,mu] if b==mu else 0)*eta[a,a]*ts[a]-(1 if a==mu else 0)*ts[b])/3))
  def trace(V):return s.Matrix([sum(V[mu][a,a]-V[a][a,mu] for a in range(4)) for mu in range(4)])
  self.assertEqual(s.simplify(trace(W)-ts),s.zeros(4,1))
  for a,b in combinations(range(4),2):
   for nu in range(4):
    delta=[s.zeros(4) for _ in range(4)];delta[nu][a,b]=eta[a,a];delta[nu][b,a]=-eta[b,b]
    dR=0
    for aa in range(4):
     for bb in range(4):
      dcomm=delta[aa]*W[bb]+W[aa]*delta[bb]-delta[bb]*W[aa]-W[bb]*delta[aa]
      dR+=eta[bb,bb]*dcomm[aa,bb]
    dsource=beta*(trace(delta).T*eta*v)[0]
    self.assertEqual(s.simplify(M2*dR/2+dsource),0)
 def test_scalar_EL_before_elimination_from_jet_action(self):
  t,x=s.symbols('t x',real=True);M2,beta=s.symbols('M2 beta',nonzero=True,real=True);a,b=s.symbols('a b',positive=True)
  p=s.Function('p')(t,x);q=s.Function('q')(t,x);tt=s.Function('tt')(t,x);tx=s.Function('tx')(t,x)
  B=(p+q)/s.sqrt(2)
  L=(s.diff(p,t)**2-s.diff(p,x)**2+s.diff(q,t)**2-s.diff(q,x)**2-a*p*p-b*q*q)/2+M2*(tt*tt-tx*tx)/3+beta*(-tt*s.diff(B,t)+tx*s.diff(B,x))
  solutions=s.solve([s.diff(L,tt),s.diff(L,tx)],[tt,tx])
  reduced=s.simplify(L.subs(solutions))
  fields=[p,q];EL=s.Matrix([s.diff(reduced,f)-s.diff(s.diff(reduced,s.diff(f,t)),t)-s.diff(s.diff(reduced,s.diff(f,x)),x) for f in fields])
  eta=3*beta*beta/(2*M2);K=s.eye(2)-eta*s.ones(2)/2
  expected=K*s.Matrix([-s.diff(p,t,2)+s.diff(p,x,2),-s.diff(q,t,2)+s.diff(q,x,2)])-s.diag(a,b)*s.Matrix(fields)
  self.assertEqual(s.simplify(EL-expected),s.zeros(2,1))
 def test_ordinary_NY_zero_but_norm_not_zero(self):
  # Pure trace form T^A=(v wedge e^A)/3 with arbitrary t=v, checked combinatorially.
  v=s.symbols('v0:4',real=True)
  def wedge(a,b):
   out={}
   for aa,x in a.items():
    for bb,y in b.items():
     seq=aa+bb
     if len(set(seq))<len(seq):continue
     key=tuple(sorted(seq));sgn=(-1)**sum(seq[i]>seq[j] for i in range(len(seq)) for j in range(i+1,len(seq)))
     out[key]=s.simplify(out.get(key,0)+sgn*x*y)
   return {k:x for k,x in out.items() if x!=0}
  one={(i,):v[i] for i in range(4)};T=[{k:x/3 for k,x in wedge(one,{(a,):1}).items()} for a in range(4)]
  eT={};TT={}
  for a in range(4):
   sign=-1 if a==0 else 1
   for dest,pair in [(eT,wedge({(a,):sign},T[a])),(TT,wedge(T[a],T[a]))]:
    for k,x in pair.items():dest[k]=s.simplify(dest.get(k,0)+x)
  self.assertTrue(all(x==0 for x in eT.values()));self.assertTrue(all(x==0 for x in TT.values()))
  # Independent component contraction of a spacelike trace, not an ordinary NY form.
  vv=s.symbols('vv',nonzero=True,real=True);tv=[0,vv,0,0];sig=[-1,1,1,1]
  norm=0
  for a in range(4):
   for m in range(4):
    for n in range(4):
     val=((1 if a==n else 0)*tv[m]-(1 if a==m else 0)*tv[n])/3
     norm+=sig[a]*sig[m]*sig[n]*val*val
  self.assertEqual(s.simplify(norm),2*vv**2/3)

class IndependentProjection(unittest.TestCase):
 def test_same_action_hessian_and_common_source_two_charts(self):
  k,a,b,z=s.symbols('k a b z',positive=True);kap=k*k;c=(1+kap)/2;h=(kap-1)/2
  # Differentiate the reduced variational density, rather than import a kernel.
  p,q=s.symbols('p q');r=s.Matrix([p,q]);K=s.Matrix([[c,h],[h,c]])
  L=(r.T*(z*K-s.diag(a,b))*r)[0]/2
  D=s.hessian(L,[p,q]);n=s.Matrix([1,0])/s.sqrt(c);J=K*n
  T1=s.Matrix([[1/s.sqrt(c),-h/(s.sqrt(c)*k)],[0,s.sqrt(c)/k]])
  T2=s.Matrix([[1/(s.sqrt(2)*k),1/s.sqrt(2)],[1/(s.sqrt(2)*k),-1/s.sqrt(2)]])
  outputs=[]
  for T in [T1,T2]:
   Dc=s.simplify(T.T*D*T);src=T.T*J
   A,B,F=Dc[0,0],Dc[0,1],Dc[1,1];E=A-B*B/F
   res=(src[0]-B*src[1]/F)**2/E+src[1]**2/F
   outputs.append(s.factor(res))
  self.assertEqual(s.simplify(outputs[0]-outputs[1]),0)
  direct=s.solve(D*r-J,[p,q]);rr=(J.T*s.Matrix([direct[p],direct[q]]))[0]
  self.assertEqual(s.simplify(rr-outputs[0]),0)
 def test_sources_not_changed_and_contact_not_optional(self):
  k,a,b,z=s.symbols('k a b z',positive=True)
  H=s.Matrix([[(a+b)/(2*k*k),(a-b)/(2*k)],[(a-b)/(2*k),(a+b)/2]])
  Dc=z*s.eye(2)-H;src=s.Matrix([k,1])/s.sqrt(1+k*k)
  A,B,F=Dc[0,0],Dc[0,1],Dc[1,1];E=A-B*B/F
  partial=(src[0]-B*src[1]/F)**2/E
  correct=(src.T*Dc.inv()*src)[0]
  self.assertEqual(s.simplify(correct-partial-src[1]**2/F),0)
  self.assertNotEqual(s.simplify(src[1]**2/F),0)
 def test_no_autonomous_species_or_bright_sector_generic(self):
  k,a,b=s.symbols('k a b',positive=True);c=(1+k*k)/2;h=(k*k-1)/2
  h1=-a*h/(c*k);h2=(a-b)/(2*k)
  self.assertEqual(s.solve(h1,k),[1])
  self.assertEqual(s.solve(h2,a),[b])
 def test_spectral_weights_positive_in_normal_form(self):
  alpha,delta,mu=s.symbols('alpha delta mu',real=True)
  gap=s.sqrt((alpha-delta)**2+4*mu*mu)
  lm=(alpha+delta-gap)/2;lp=(alpha+delta+gap)/2
  wl=(delta-lm)/gap;wh=(lp-delta)/gap
  self.assertEqual(s.simplify(wl+wh),1)
  # The exact identity implies strict positive weights when mu!=0.
  self.assertEqual(s.simplify(wl*wh-mu*mu/gap**2),0)
 def test_block_chart_zero_is_not_physical_pole(self):
  k,a,b=s.symbols('k a b',positive=True);z=s.symbols('z');kap=k*k;c=(1+kap)/2;h=(kap-1)/2
  det=kap*z*z-c*(a+b)*z+a*b
  self.assertEqual(s.factor(det.subs(z,(a+b)/2)),-(a-b)**2/4)
  # A physical spectrum is not determined by this arbitrary complementary entry.
  self.assertEqual(s.simplify(det.subs(z,(a*h*h/(kap*c)+b*c/kap))+a*a*(kap-1)**2/(kap+1)**2),0)
 def test_critical_locus_has_constraint_not_two_wave_modes(self):
  a,b,z,B,D,Jb,Jd=s.symbols('a b z B D Jb Jd',positive=True)
  m=(a+b)/2;d=(a-b)/2
  Bstar=s.solve(-m*B-d*D-Jb,B)[0]
  eq=s.expand((z-m)*D-d*Bstar-Jd)
  self.assertEqual(s.simplify(s.diff(eq,D)-(z-2*a*b/(a+b))),0)
 def test_covariant_principal_cones_from_kinetic_eigenbasis(self):
  kap,z=s.symbols('kap z');U=s.Matrix([[1,1],[1,-1]])/s.sqrt(2)
  K=s.eye(2)+(kap-1)*s.ones(2)/2
  self.assertEqual(s.simplify(U.T*K*U),s.diag(kap,1))
  self.assertEqual((z*K).det(),kap*z*z)
 def test_pt_retarded_to_advanced(self):
  t,om=s.symbols('t om',real=True,nonzero=True)
  ret=s.Heaviside(t)*s.sin(om*t)/om
  adv=-s.Heaviside(-t)*s.sin(om*t)/om
  self.assertEqual(s.simplify(ret.subs(t,-t)-adv),0)

if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 Path(__file__).with_name('independent_test_results.json').write_text(json.dumps({'sympy':s.__version__,'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'success':result.wasSuccessful(),'independence':'no import of derivation scripts/results; direct 24 variation test, action-jet EL and matched-source chart solve'},indent=2))
 raise SystemExit(0 if result.wasSuccessful() else 1)
