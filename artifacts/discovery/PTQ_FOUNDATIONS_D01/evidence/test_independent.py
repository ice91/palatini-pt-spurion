"""Independent checks: no import of derive_models.py or its output formulas."""
from pathlib import Path
import itertools,json,unittest
import sympy as s
X=s.symbols('t x y z',real=True)

def hsimp(expr):
 return s.simplify(s.expand(expr.rewrite(s.exp)))

def simp(f):return {k:s.simplify(v) for k,v in f.items() if s.simplify(v)!=0}
def add(*fs):
 o={}
 for f in fs:
  for k,v in f.items():o[k]=o.get(k,0)+v
 return simp(o)
def sc(f,c):return simp({k:c*v for k,v in f.items()})
def wedge(a,b):
 o={}
 for ia,va in a.items():
  for ib,vb in b.items():
   word=ia+ib
   if len(set(word))!=len(word):continue
   sign=(-1)**sum(word[i]>word[j] for i in range(len(word)) for j in range(i+1,len(word)))
   key=tuple(sorted(word));o[key]=o.get(key,0)+sign*va*vb
 return simp(o)
def d(f):
 out={}
 for inds,v in f.items():
  for j in range(4):out=add(out,wedge({(j,):s.diff(v,X[j])},{inds:1}))
 return out
def U(f):
 sub={x:-x for x in X}
 return simp({inds:(-1)**len(inds)*s.conjugate(v).subs(sub,simultaneous=True) for inds,v in f.items()})
def star(f):
 out={}
 for inds,v in f.items():
  comp=tuple(i for i in range(4) if i not in inds);word=inds+comp
  sign=(-1)**sum(word[i]>word[j] for i in range(4) for j in range(i+1,4))
  out[comp]=out.get(comp,0)+v*sign*(-1 if 0 in inds else 1)
 return simp(out)

class GeometryChecks(unittest.TestCase):
 def test_natural_lift_d_wedge_and_involution(self):
  t,x,y,z=X
  f={(0,):s.I*t*x,(2,):y*y+z};h={(1,):t+y,(3,):s.I*z}
  self.assertEqual(U(U(f)),simp(f))
  self.assertEqual(U(d(f)),d(U(f)))
  self.assertEqual(U(wedge(f,h)),wedge(U(f),U(h)))
 def test_cartan_from_compatible_connection(self):
  t,x,y,z=X
  e=[{(0,):1+t*t},{(1,):1+x*x},{(2,):1+y*y},{(3,):1+z*z}]
  e[1]=add(e[1],{(2,):t*x})
  # Lowered Lorentz pair antisymmetric; raising first index preserves compatibility.
  lower=[[{} for _ in range(4)] for _ in range(4)]
  lower[0][1]={(2,):t+x};lower[1][0]=sc(lower[0][1],-1)
  lower[1][2]={(0,):y*z};lower[2][1]=sc(lower[1][2],-1)
  w=[[sc(lower[a][b],-1 if a==0 else 1) for b in range(4)] for a in range(4)]
  def tors(e,w):return [add(d(e[a]),*[wedge(w[a][b],e[b]) for b in range(4)]) for a in range(4)]
  def curv(w):return [[add(d(w[a][b]),*[wedge(w[a][c],w[c][b]) for c in range(4)]) for b in range(4)] for a in range(4)]
  up=[[U(f) for f in row] for row in w]
  self.assertEqual([U(f) for f in tors(e,w)],tors([U(f) for f in e],up))
  self.assertEqual([[U(f) for f in row] for row in curv(w)],curv(up))
 def test_hodge_minkowski(self):
  t,x,y,z=X
  f={(0,1):s.I*x*y,(1,3):t*t+z}
  self.assertEqual(U(star(f)),star(U(f)))
  self.assertEqual(star(star(f)),sc(f,-1))
 def test_coframe_orientation_and_time_orientation(self):
  self.assertEqual((-s.eye(4)).det(),1)
  self.assertEqual(U({(0,):1}),{(0,):-1})

class IndependentDynamicsChecks(unittest.TestCase):
 def test_metric_and_real_action_from_hyperbolic_similarity(self):
  c,d,h=s.symbols('c d h',real=True)
  sy=s.Matrix([[0,-s.I],[s.I,0]])
  O=s.cosh(h/2)*s.eye(2)+s.sinh(h/2)*sy
  Oinv=s.cosh(h/2)*s.eye(2)-s.sinh(h/2)*sy
  diag=s.diag(c-d,c+d)
  # Reconstruct M from ordinary healthy control, independent of a formula oracle.
  M=(Oinv*diag*O).applyfunc(hsimp)
  Mexpected=s.Matrix([[c-d*s.cosh(h),s.I*d*s.sinh(h)],[s.I*d*s.sinh(h),c+d*s.cosh(h)]])
  self.assertEqual((M-Mexpected).applyfunc(hsimp),s.zeros(2))
  G=(O.conjugate().T*O/s.cosh(h)).applyfunc(hsimp)
  self.assertEqual((M.conjugate().T*G-G*M).applyfunc(hsimp),s.zeros(2))
  self.assertEqual((s.diag(1,-1)*G.conjugate()*s.diag(1,-1)-G).applyfunc(hsimp),s.zeros(2))
 def test_response_via_normalized_source_in_control_basis(self):
  h,c,d,z=s.symbols('h c d z',real=True)
  # Compute weights from normalized source, not from a Schur numerator.
  w1=s.cosh(h/2)**2/s.cosh(h);w2=s.sinh(h/2)**2/s.cosh(h)
  self.assertEqual(hsimp(w1+w2),1)
  self.assertEqual(hsimp(w1-(1+1/s.cosh(h))/2),0)
  self.assertEqual(hsimp(w2-(1-1/s.cosh(h))/2),0)
  sy=s.Matrix([[0,-s.I],[s.I,0]])
  O=(s.cosh(h/2)*s.eye(2)+s.sinh(h/2)*sy)/s.sqrt(s.cosh(h))
  e=s.Matrix([1,0]); q=O*e
  computed=(q.conjugate().T*s.diag(1/(z-c+d),1/(z-c-d))*q)[0]
  self.assertEqual(hsimp(computed-w1/(z-c+d)-w2/(z-c-d)),0)
 def test_actual_physical_action_elimination_with_source_dressing(self):
  a,b,g,z=s.symbols('a b g z',real=True);r=2*g/(b-a)
  G=s.Matrix([[1,-s.I*r],[s.I*r,1]])
  # Differentiate a quadratic variational density with independent conjugate jets.
  q=s.Matrix(s.symbols('p q'));bar=s.Matrix(s.symbols('pb qb'))
  M=s.Matrix([[a,s.I*g],[s.I*g,b]])
  density=(bar.T*G*(z*s.eye(2)-M)*q)[0]
  H=s.Matrix([[s.diff(density,bar[i],q[j]) for j in range(2)] for i in range(2)])
  e=s.Matrix([1,0]);v=G*e
  qsol=s.solve(H*s.Matrix(s.symbols('p q'))-v,list(q))
  response=s.factor((e.conjugate().T*G*s.Matrix([qsol[q[0]],qsol[q[1]]]))[0])
  # Independently solve equations in components and include dressed sources/contact.
  F=H[1,1];heff=H[0,0]-H[0,1]*H[1,0]/F
  vr=v[0]-H[0,1]*v[1]/F;vl=s.conjugate(v[0])-s.conjugate(v[1])*H[1,0]/F
  self.assertEqual(s.simplify(vl*vr/heff+s.conjugate(v[1])*v[1]/F-response),0)
  self.assertEqual(s.simplify(response.subs(z,b)-2/(b-a)),0)
 def test_exceptional_point_and_broken_counterexample(self):
  c,d,z=s.symbols('c d z',positive=True)
  M=s.Matrix([[c-d,s.I*d],[s.I*d,c+d]])
  N=M-c*s.eye(2)
  self.assertEqual(N*N,s.zeros(2));self.assertEqual(N.rank(),1)
  broken=s.Matrix([[c-d,2*s.I*d],[2*s.I*d,c+d]])
  self.assertEqual(s.expand((z*s.eye(2)-broken).det()),s.expand((z-c)**2+3*d*d))
 def test_same_cone_even_in_broken_parameter_region(self):
  z,a,b,g=s.symbols('z a b g',real=True)
  eps=s.symbols('eps')
  kernel=s.Matrix([[z-eps*a,-s.I*eps*g],[-s.I*eps*g,z-eps*b]])
  self.assertEqual(kernel.det().subs(eps,0),z*z)
 def test_EFT_truncation_has_spurious_out_of_window_root(self):
  eps,z=s.symbols('eps z',positive=True)
  # Dimensionless b=1 and a=eps: a controlled scale limit, not a data fit.
  exact=(z-eps)*(z-1)+eps**2
  trunc=-eps**2*z*z+(1-eps**2)*z-eps-eps**2
  roots=s.solve(trunc,z)
  large=[r for r in roots if s.limit(eps**2*r,eps,0)==1]
  self.assertEqual(len(large),1)
  self.assertEqual(s.limit(exact.subs(z,large[0])*eps**4,eps,0),1)
 def test_pt_sends_retarded_oscillator_green_to_advanced(self):
  t,w=s.symbols('t w',real=True,nonzero=True)
  ret=s.Heaviside(t)*s.sin(w*t)/w
  adv=-s.Heaviside(-t)*s.sin(w*t)/w
  self.assertEqual(s.simplify(ret.subs(t,-t)-adv),0)
 def test_generic_product_identity_not_multiplicativity(self):
  xp,xm,yp,ym=s.symbols('xp xm yp ym',commutative=False)
  prod=s.expand((xp+xm)*(yp+ym));trans=s.expand((xp-xm)*(yp-ym))
  self.assertEqual(s.expand((prod+trans)/2-xp*yp-xm*ym),0)

if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 Path(__file__).with_name('independent_test_results.json').write_text(json.dumps({'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'success':result.wasSuccessful(),'sympy_version':s.__version__},indent=2))
 raise SystemExit(0 if result.wasSuccessful() else 1)
