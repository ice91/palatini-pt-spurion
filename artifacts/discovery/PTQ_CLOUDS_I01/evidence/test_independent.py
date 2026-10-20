"""Independent Fourier quadrature, finite Fock and channel calculations.

No import of derive_observer.py or its saved results. All numerical examples
are dimensionless consistency checks, not empirical fits.
"""
from pathlib import Path
import json,unittest
import mpmath as mp
import sympy as S
HERE=Path(__file__).resolve().parent
mp.mp.dps=45
records={}

def rate(e):
    return e/mp.expm1(2*mp.pi*e) if e else 1/(2*mp.pi)

class IndependentChecks(unittest.TestCase):
    def test_contour_fourier_response(self):
        rows=[]
        for e in [mp.mpf('-1'),mp.mpf('-.25'),mp.mpf('0'),mp.mpf('.25'),mp.mpf('1')]:
            for fraction in [mp.mpf(1)/3,mp.mpf(1)/2]:
                delta=2*mp.pi*fraction
                def f(t):
                    z=t-mp.j*delta
                    return -mp.exp(-mp.j*e*z)/(8*mp.pi*mp.sinh(z/2)**2)
                val=mp.quad(f,[-mp.inf,-8,-2,0,2,8,mp.inf])
                error=abs(val-rate(e))
                self.assertLess(error,mp.mpf('1e-35'))
                rows.append({'E_over_a':str(e),'strip_fraction':str(fraction),
                             'quadrature_real':str(val.real),'imaginary':str(val.imag),
                             'absolute_error':str(error)})
        records['fourier_quadratures']=rows

    def test_smooth_switching_and_stationary_limit(self):
        # chi=exp[-tau^2/(2T^2)], F_chi / integral chi^2.
        # Change nu=E+x/T for stable integration on fixed x domain.
        rows=[]; e=mp.mpf('0.5')
        for T in [2,5,15,40]:
            val=mp.quad(lambda x:mp.exp(-x*x)*rate(e+x/T)/mp.sqrt(mp.pi),
                        [-mp.inf,-T*e,0,mp.inf])
            inertial=mp.quad(lambda x:mp.exp(-x*x)*max(-(e+x/T),0)/mp.sqrt(mp.pi),
                             [-mp.inf,-T*e,0,mp.inf])
            self.assertGreater(val,0);self.assertGreater(inertial,0)
            rows.append({'duration_aT':T,'accelerated_normalized_response':str(val),
                         'inertial_switching_excitation':str(inertial),
                         'stationary_target':str(rate(e))})
        self.assertLess(abs(mp.mpf(rows[-1]['accelerated_normalized_response'])-rate(e)),mp.mpf('.0001'))
        records['switching']=rows

    def test_fock_sums_without_density_kernel(self):
        q=mp.mpf(1)/2
        probs=[(1-q*q)*q**(2*n) for n in range(160)]
        nbar=sum(n*p for n,p in enumerate(probs))
        Snum=-sum(p*mp.log(p) for p in probs)
        purity=sum(p*p for p in probs)
        cross=(1-q*q)*sum((n+1)*q**(2*n+1) for n in range(160))
        self.assertAlmostEqual(float(nbar),1/3,places=14)
        self.assertLess(abs(Snum-(-mp.log(1-q*q)-q*q/(1-q*q)*mp.log(q*q))),mp.mpf('1e-40'))
        self.assertLess(abs(purity-(1-q*q)/(1+q*q)),mp.mpf('1e-40'))
        self.assertLess(abs(cross-q/(1-q*q)),mp.mpf('1e-40'))
        records['mode_information']={'q':'1/2','entropy_nats':str(Snum),
                                    'mean_number':str(nbar),'purity':str(purity),
                                    'cross_correlation':str(cross)}

    def test_gaussian_marginal_by_actual_integration(self):
        q=mp.mpf(1)/2; A=(1+q*q)/(1-q*q); C=2*q/(1-q*q)
        # det precision=1, normalized two-oscillator wave function.
        psi=lambda x,y:mp.exp(-(A*(x*x+y*y)-2*C*x*y)/2)/mp.sqrt(mp.pi)
        for x,xp in [(mp.mpf(0),mp.mpf('.3')),(mp.mpf('.4'),mp.mpf('-.2'))]:
            actual=mp.quad(lambda y:psi(x,y)*psi(xp,y),[-mp.inf,0,mp.inf])
            expected=mp.exp(-A*(x*x+xp*xp)/2+C*C*(x+xp)**2/(4*A))/mp.sqrt(mp.pi*A)
            self.assertLess(abs(actual-expected),mp.mpf('1e-40'))
        # Momentum variance obtained by integrating |d_x psi|^2 analytically
        # with covariance inverse of [[A,-C],[-C,A]]/2, not saved formulas.
        K=S.Matrix([[S.Rational(5,3),-S.Rational(4,3)],[-S.Rational(4,3),S.Rational(5,3)]])
        cov=K.inv()/2; grad=S.Matrix([K[0,0],K[0,1]])
        self.assertEqual((grad.T*cov*grad)[0],S.Rational(5,6))
        self.assertEqual(cov[0,0],S.Rational(5,6))

    def test_channel_on_ancilla_and_hilbert_postselection(self):
        # Input R,L each dimension2, entangled with ancillary A. Test extension
        # explicitly in addition to the exact Kraus/Choi proof in main script.
        d=2; v=S.Matrix([1,2,0,1,3,0,2,1]); rho=v*v.T/(v.T*v)[0]
        red=S.zeros(4)
        for i in range(2):
            for j in range(2):
                for a in range(2):
                    for b in range(2):
                        red[2*i+a,2*j+b]=sum(rho[4*i+2*l+a,4*j+2*l+b] for l in range(2))
        self.assertEqual(S.trace(red),1)
        for k in range(1,5):
            # all principal minors, not just leading ones
            import itertools
            for inds in itertools.combinations(range(4),k):
                self.assertGreaterEqual(red.extract(inds,inds).det(),0)
        # Entangled pair |00>+q|11>: marginal and conditional output differ.
        q=S.Rational(1,2); psi=S.Matrix([1,0,0,q])/S.sqrt(1+q*q)
        pair=psi*psi.T
        reduced=S.diag(pair[0,0],pair[3,3])
        conditional=S.diag(pair[0,0],0)
        self.assertEqual(S.trace(reduced),1)
        self.assertEqual(S.trace(conditional),S.Rational(4,5))
        self.assertNotEqual(reduced,conditional/S.trace(conditional))

    def test_same_marginals_different_global_correlations(self):
        q=S.Rational(1,2);psi=S.Matrix([1,0,0,q])/S.sqrt(1+q*q)
        pure=psi*psi.T;dephased=S.diag(*pure.diagonal())
        b=S.Matrix([[0,1],[0,0]]);cross=S.kronecker_product(b,b)
        self.assertEqual(S.trace(cross*pure),S.Rational(2,5))
        self.assertEqual(S.trace(cross*dephased),0)
        self.assertEqual(S.trace(pure*pure),1)
        self.assertEqual(S.trace(dephased*dephased),S.Rational(17,25))

    def test_retarded_advanced_and_spacelike_support(self):
        def ret(t,x):return S.Rational(1,2) if t>0 and t*t>x*x else S.Integer(0)
        def adv(t,x):return S.Rational(1,2) if t<0 and t*t>x*x else S.Integer(0)
        for t,x in [(3,1),(-3,1),(1,3),(-1,3)]:
            self.assertEqual(ret(-t,x),adv(t,x))
            if x*x>t*t:self.assertEqual(ret(t,x)-adv(t,x),0)
        # Any point in R and any in L are spacelike separated.
        ur,vr,ul,vl=-S.Rational(2,3),S.Rational(3,2),S.Rational(5,4),-S.Rational(4,5)
        self.assertLess((ur-ul)*(vr-vl),0)

    def test_time_reversal_no_monotone_entropy_and_stationarity(self):
        # A finite regulated wedge is diagonal in N for every inverse temperature;
        # all its phases cancel. No entropy increase even with thermal marginal.
        beta,w,t=S.symbols('beta w t',real=True)
        p=S.diag(1,S.exp(-beta*w))/(1+S.exp(-beta*w))
        V=S.diag(1,S.exp(-S.I*w*t))
        self.assertEqual(S.simplify(V*p*V.conjugate().T-p),S.zeros(2))
        # If both f'(t)>=0 and reversed trajectory derivative -f'(-t)>=0
        # are demanded for every allowed trajectory, only f'=0 can satisfy both.
        records['no_go_scope']='analytic universal monotonicity theorem in report03; stationarity verified here'

    def test_counterstate_and_noise_response_identity(self):
        # Integrate the positive-frequency wavepacket rather than assume its
        # position-space form. Work in units a=ell=1.
        rows=[]
        for tau in [mp.mpf(0),mp.log(2)]:
            U=-mp.exp(-tau);Ud=mp.exp(-tau)
            amp=mp.quad(lambda k:-mp.j*k*Ud*mp.exp(-k)*mp.exp(-mp.j*k*U)/mp.sqrt(mp.pi),[0,2,8,mp.inf])
            excess=2*abs(amp)**2
            expected=2*mp.exp(-2*tau)/(mp.pi*(1+mp.exp(-2*tau))**2)
            self.assertLess(abs(excess-expected),mp.mpf('1e-40'))
            rows.append({'tau':str(tau),'excess_equal_time_derivative_variance':str(excess)})
        self.assertNotEqual(rows[0]['excess_equal_time_derivative_variance'],rows[1]['excess_equal_time_derivative_variance'])
        records['counterstate']=rows
        for e in [mp.mpf('.25'),mp.mpf(1)]:
            self.assertLess(abs(rate(-e)-rate(e)-e),mp.mpf('1e-40'))
            self.assertLess(abs(rate(-e)+rate(e)-e/mp.tanh(mp.pi*e)),mp.mpf('1e-40'))

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(IndependentChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    payload={'success':result.wasSuccessful(),'tests':result.testsRun,
             'failures':len(result.failures),'errors':len(result.errors),
             'mpmath_precision':mp.mp.dps,'records':records}
    (HERE/'independent_results.json').write_text(json.dumps(payload,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
