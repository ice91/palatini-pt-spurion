"""Independent I02 checks: no imports of derivation scripts or result files.
Finite matrices certify their domains, not a continuum entropy cutoff.
"""
from pathlib import Path
import json,unittest,itertools
import sympy as S
import mpmath as mp
HERE=Path(__file__).resolve().parent
mp.mp.dps=50
records={}

def entropy(mat):
    eig=mp.eigsy(mat,eigvals_only=True)
    assert all(p>0 for p in eig)
    return -sum(p*mp.log(p) for p in eig)
sigma=mp.diag([mp.mpf(16)/21,mp.mpf(4)/21,mp.mpf(1)/21])
X=mp.matrix([[-1,1,0],[1,1,0],[0,0,0]])
K=mp.diag([-mp.log(sigma[i,i]) for i in range(3)])
def trace(m):return sum(m[i,i] for i in range(m.rows))
def rel(r):return -entropy(r)+trace(r*K)

def bump(y):
    return mp.exp(-1/(1-y*y)) if abs(y)<1 else mp.mpf(0)

class IndependentChecks(unittest.TestCase):
    def test_noncommuting_first_law_and_hessian(self):
        Sdot=mp.diff(lambda eps:entropy(sigma+eps*X),mp.mpf(0))
        Kdot=trace(X*K)
        D2=mp.diff(lambda eps:rel(sigma+eps*X),mp.mpf(0),2)
        target=mp.mpf(105)/16+mp.mpf(7)/2*mp.log(4)
        self.assertLess(abs(Sdot-Kdot),mp.mpf('1e-40'))
        self.assertLess(abs(D2-target),mp.mpf('1e-35'))
        self.assertGreater(D2,0)
        records['first_law']={'entropy_derivative':str(Sdot),'modular_energy_derivative':str(Kdot),'D_second_derivative':str(D2)}

    def test_relative_entropy_and_access_monotonicity(self):
        rows=[]
        for eps in [mp.mpf('-0.02'),mp.mpf('0.02')]:
            rho=sigma+eps*X
            pinched=mp.diag([rho[i,i] for i in range(3)])
            d=rel(rho);dc=rel(pinched)
            self.assertGreater(d,dc);self.assertGreater(dc,0)
            lhs=trace((rho-sigma)*K)-(entropy(rho)-entropy(sigma))
            self.assertLess(abs(d-lhs),mp.mpf('1e-45'))
            rows.append({'epsilon':str(eps),'D_quantum':str(d),'D_number_measurement':str(dc),
                         'deltaK':str(trace((rho-sigma)*K)),'deltaS':str(entropy(rho)-entropy(sigma))})
        records['relative_entropy']=rows

    def test_unitary_change_has_entropy_zero_energy_nonzero(self):
        angle=mp.mpf('0.3');c,s=mp.cos(angle),mp.sin(angle)
        V=mp.matrix([[c,-s,0],[s,c,0],[0,0,1]])
        r=V*sigma*V.T
        dk=trace((r-sigma)*K)
        self.assertLess(abs(entropy(r)-entropy(sigma)),mp.mpf('1e-45'))
        self.assertLess(abs(rel(r)-dk),mp.mpf('1e-45'))
        self.assertGreater(dk,0)
        records['unitary_counterexample']={'deltaS':'0 (unitary spectrum invariance)','deltaK':str(dk),'D':str(rel(r))}

    def test_compact_coherent_stress_and_modular_cocycle(self):
        # Different coordinate variable and numerical differentiation of the
        # profile, rather than the derivative formula in derive_modular.py.
        d=mp.mpf(3);L=mp.mpf(1);lam=mp.mpf(1)/5
        f=lambda x:lam*bump((x-d)/L)
        fd=lambda x:mp.diff(f,x)
        h=lambda x:fd(x)**2/2
        cuts=[d-L,d,d+L]
        energy=mp.quad(h,cuts)
        dk=2*mp.pi*mp.quad(lambda x:x*h(x),cuts)
        self.assertLess(abs(dk-2*mp.pi*d*energy),mp.mpf('1e-40'))
        # Symplectic cocycle: gdot_s=2pi*(x f')'. Its phase derivative
        # determines D=-1/2*symplectic(v,vdot), independently of h(x).
        sympl=mp.quad(lambda x:f(x)*2*mp.pi*mp.diff(lambda y:y*fd(y),x),cuts)
        self.assertLess(abs(-sympl/2-dk),mp.mpf('1e-40'))
        # Nested translated wedge, packet still wholly contained in both.
        b=mp.mpf('0.5')
        db=2*mp.pi*mp.quad(lambda x:(x-b)*h(x),cuts)
        self.assertLess(db,dk)
        self.assertLess(abs(dk-db-2*mp.pi*b*energy),mp.mpf('1e-40'))
        records['coherent_packet']={'energy':str(energy),'D_right':str(dk),
                                    'minus_half_symplectic_derivative':str(-sympl/2),
                                    'D_shifted_wedge_half':str(db)}

    def test_coherent_wave_stress_conservation(self):
        # Local jet proof independent of the symmetric two-wave expression.
        pt,px,ptt,ptx,pxx=S.symbols('pt px ptt ptx pxx',real=True)
        h=(pt*pt+px*px)/2;j=-pt*px
        div=S.diff(h,pt)*ptt+S.diff(h,px)*ptx+S.diff(j,pt)*ptx+S.diff(j,px)*pxx
        self.assertEqual(S.expand(div-pt*(ptt-pxx)),0)
        # +/- coherent means have identical stress but different current.
        f0,der=S.symbols('f0 der',real=True,nonzero=True)
        self.assertEqual(S.simplify(der**2-(-der)**2),0)
        self.assertNotEqual(der,-der)

    def test_two_dimensional_curvature_obstruction(self):
        # Tensor identity in any 2D Lorentz frame, independent of coordinate
        # Christoffel derivation: only one Riemann component is independent.
        eta=S.diag(1,-1);inv=eta.inv();r=S.symbols('r')
        riem=lambda a,b,c,d:r*(eta[a,c]*eta[b,d]-eta[a,d]*eta[b,c])/2
        Ric=S.Matrix(2,2,lambda b,d:sum(inv[a,c]*riem(a,b,c,d) for a in range(2) for c in range(2)))
        self.assertEqual(Ric,eta*r/2)
        scalar=sum(inv[a,b]*Ric[a,b] for a in range(2) for b in range(2))
        self.assertEqual(Ric-eta*scalar/2,S.zeros(2))

    def test_conditional_horizon_coefficient_and_null_lemma(self):
        # Given the ADDED Clausius/area premises, integrate the same patch in
        # a different variable u=-lambda_affine. Do not assume Einstein eq.
        u,ell,k,alpha,T=S.symbols('u ell k alpha T',positive=True)
        R=S.symbols('R',real=True)
        lhs=S.integrate(alpha*u*R,(u,0,ell))
        rhs=S.integrate(k*u*T,(u,0,ell))/(k/(2*S.pi))
        self.assertEqual(S.solve(lhs-rhs,R)[0],2*S.pi*T/alpha)
        # Explicit 3D tensor solve for five independent null directions.
        a,b,c,d,e,f=S.symbols('a b c d e f')
        M=S.Matrix([[a,b,c],[b,d,e],[c,e,f]])
        ns=[S.Matrix(v) for v in [(1,1,0),(1,-1,0),(1,0,1),(1,0,-1),(1,S.sqrt(2)/2,S.sqrt(2)/2)]]
        sol=S.solve([(n.T*M*n)[0] for n in ns],[b,c,d,e,f])
        self.assertEqual(M.subs(sol),S.diag(a,-a,-a))

    def test_detector_excess_by_frequency_convolution(self):
        # Independent of the time-integral implementation. Analytic unwindowed
        # wave-packet amplitude is convolved with the known Gaussian switch.
        E=mp.mpf('0.5')
        def A(nu):
            return mp.sqrt(mp.pi)*nu*mp.exp(-mp.pi*nu/2)/mp.sinh(mp.pi*nu) if nu else 1/mp.sqrt(mp.pi)
        rows=[]
        for T in [mp.mpf(2),mp.mpf(5)]:
            def conv(e):
                return mp.quad(lambda y:mp.exp(-y*y/2)*A(e+y/T),[-mp.inf,-T*e,0,mp.inf])/mp.sqrt(2*mp.pi)
            ip,im=conv(E),conv(-E)
            # Independently integrate real half-line form of Fdot.
            def real_time(e):
                return mp.quad(lambda t:mp.exp(-t*t/(2*T*T))*(mp.cos(e*t)-mp.sinh(t)*mp.sin(e*t))/mp.cosh(t)**2,
                               [0,2,8,mp.inf])/mp.sqrt(mp.pi)
            self.assertLess(abs(ip-real_time(E)),mp.mpf('1e-35'))
            self.assertLess(abs(im-real_time(-E)),mp.mpf('1e-35'))
            excess=ip**2+im**2
            self.assertGreater(excess,0)
            rows.append({'T':str(T),'I_E':str(ip),'I_minusE':str(im),'state_excess':str(excess)})
        records['detector_independent_convolution']=rows

    def test_identical_local_restrictions_no_protocol_distinguishes(self):
        # Concrete entangled/dephased I01 pair and a nonprojective POVM. The
        # general algebra proof for all instruments is displayed in report03.
        q=S.Rational(1,2);v=S.Matrix([1,0,0,q])/S.sqrt(1+q*q)
        pure=v*v.T;deph=S.diag(*pure.diagonal())
        E=S.Matrix([[S.Rational(1,3),S.Rational(1,6)],[S.Rational(1,6),S.Rational(2,3)]])
        for F in [E,S.eye(2)-E]:
            self.assertEqual(S.trace(S.kronecker_product(F,S.eye(2))*pure),S.trace(S.kronecker_product(F,S.eye(2))*deph))
        # The same field's coherent +/- means can share quadratic stress
        # without sharing all local expectations: no claim of total blindness.

    def test_one_particle_energy_without_inferring_relative_entropy(self):
        # Energy from the momentum-space preparation, independently compare
        # with position-space stress and its finite modular-energy moment.
        ell=mp.mpf('1.3')
        energy_k=mp.quad(lambda k:4*ell**2*k*k*mp.exp(-2*ell*k),[0,1,mp.inf])
        stress=lambda x:2*ell**2/(mp.pi*(ell**2+x*x)**2)
        energy_x=mp.quad(stress,[-mp.inf,0,mp.inf])
        dk=2*mp.pi*mp.quad(lambda x:x*stress(x),[0,ell,mp.inf])
        self.assertLess(abs(energy_k-energy_x),mp.mpf('1e-40'))
        self.assertLess(abs(dk-2),mp.mpf('1e-40'))
        records['one_particle_modular_energy']={'ell':str(ell),'total_energy':str(energy_k),'right_deltaK':str(dk),
            'warning':'No Araki D or continuum entropy difference inferred from this number.'}

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentChecks))
    out={'success':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),
         'errors':len(result.errors),'precision_digits':mp.mp.dps,'records':records}
    (HERE/'independent_results.json').write_text(json.dumps(out,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
