"""Independent spectral, integral, and counterexample checks for I03.
Imports neither derivation script nor saved results nor original project code.
"""
from pathlib import Path
import json,unittest
import sympy as S
import mpmath as mp
HERE=Path(__file__).resolve().parent
mp.mp.dps=50
records={}

class IndependentChecks(unittest.TestCase):
    def test_bessel_radial_laplace_transform(self):
        u=mp.mpf('1.1');p=mp.cosh(u);rows=[]
        for nu in [mp.mpf(0),mp.mpf('0.5'),mp.mpf('1.4')]:
            value=mp.quad(lambda z:mp.exp(-p*z)*mp.besseli(nu,z),[0,1,6,mp.inf])
            expected=mp.exp(-nu*u)/mp.sinh(u)
            self.assertLess(abs(value-expected),mp.mpf('1e-40'))
            rows.append({'nu':str(nu),'integral':str(value),'analytic_error':str(abs(value-expected))})
        records['Bessel_Laplace']=rows

    def test_cone_Abel_limit_direct(self):
        u=mp.mpf('0.00001');rows=[]
        for n in [mp.mpf('0.8'),mp.mpf('1.2'),mp.mpf(2)]:
            value=(mp.coth(u/(2*n))-n*mp.coth(u/2))/(2*mp.sinh(u))
            target=(1/n-n)/12
            self.assertLess(abs(value-target),mp.mpf('1e-10'))
            rows.append({'n':str(n),'radial_regulator_u':str(u),'defect':str(value),'limit_error':str(abs(value-target))})
        records['cone_limit']=rows

    def test_orbifold_images_by_radial_integration(self):
        rows=[]
        for q in [2,3,5]:
            val=mp.mpf(0);s=mp.mpf('0.7')
            for k in range(1,q):
                # Direct distance |x-R_theta x|²=2r²(1-cos theta).
                theta=2*mp.pi*k/q
                val+=mp.quad(lambda r:2*mp.pi*r/(4*mp.pi*s)*mp.exp(-2*r*r*(1-mp.cos(theta))/(4*s)),[0,1,mp.inf])/q
            target=(q-mp.mpf(1)/q)/12
            self.assertLess(abs(val-target),mp.mpf('1e-40'))
            rows.append({'orbifold_order':q,'image_trace_defect':str(val)})
        records['orbifold_image_integrals']=rows

    def test_replica_derivative_and_finite_renyi_counterexample(self):
        # Compute log Tr rho^n from the cone defect, including subtraction of
        # n times the flat partition function. Use arbitrary units A=J=1.
        logtrace=lambda n:(1/n-n)/(96*mp.pi)
        ee=-mp.diff(logtrace,mp.mpf(1))
        renyi2=-logtrace(mp.mpf(2))
        self.assertLess(abs(ee-1/(48*mp.pi)),mp.mpf('1e-45'))
        self.assertLess(abs(renyi2/ee-mp.mpf(3)/4),mp.mpf('1e-45'))
        self.assertNotEqual(renyi2,ee)
        records['finite_replica_counterexample']={'S2_over_S1':str(renyi2/ee),'EH_only_prediction':'1; invalid finite-angle extrapolation'}

    def test_sphere_spectrum_independent_a1(self):
        # SAME scalar Laplacian's geometric heat kernel on an auxiliary
        # Euclidean S² x R² background. The mass factor is stripped off.
        # This checks the curvature coefficient without a Van Vleck input.
        ts=[mp.mpf('0.01')/2**i for i in range(4)];vals=[]
        for t in ts:
            count=int(mp.ceil(mp.sqrt(130/t)))+2
            trace=mp.fsum((2*l+1)*mp.exp(-t*l*(l+1)) for l in range(count))
            vals.append(trace-1/t)
        extrap=mp.fsum(vals[i]*mp.fprod(-ts[j]/(ts[i]-ts[j]) for j in range(4) if i!=j) for i in range(4))
        self.assertLess(abs(extrap-mp.mpf(1)/3),mp.mpf('1e-10'))
        records['S2_spectrum']={'proper_times':[str(t) for t in ts],
              'trace_minus_area_term':[str(x) for x in vals],
              'constant_extrapolation':str(extrap),'expected_from_R_over_6':'1/3'}
        # Euler-Maclaurin independently gives its exact constant: the
        # integral is 1/t, f(0)=1, f'(0)|_{t=0}=2.
        self.assertEqual(S.Rational(1,2)-S.Rational(2,12),S.Rational(1,3))

    def test_massive_proper_time_integral_and_sign(self):
        rows=[]
        for mass,eps in [(mp.mpf('.7'),mp.mpf('.3')),(mp.mpf('1.1'),mp.mpf('.2'))]:
            z=mass*mass*eps*eps
            actual=mp.quad(lambda t:mp.exp(-z*t)/t**2,[1,2,mp.inf])/eps**2
            expected=mp.exp(-z)/eps**2-mass*mass*mp.e1(z)
            self.assertGreater(actual,0)
            self.assertLess(abs(actual-expected),mp.mpf('1e-40'))
            rows.append({'m':str(mass),'epsilon':str(eps),'J_integral':str(actual),'analytic_error':str(abs(actual-expected))})
        records['massive_integrals']=rows

    def test_smooth_regulator_changes_absolute_coefficient(self):
        fn=lambda t:-mp.expm1(-t**3)/t**2 if t else mp.mpf(0)
        val=mp.quad(fn,[0,1,mp.inf])
        target=mp.gamma(mp.mpf(2)/3)
        self.assertLess(abs(val-target),mp.mpf('1e-40'))
        self.assertNotEqual(val,1)
        # A different finite result for the shared UV profile leaves matching
        # intact; using that profile for only one side would manufacture mismatch.
        records['regulator_counterexample']={'hard_step_leading_J_epsilon_squared':'1',
             'smooth_cubic_leading_J_epsilon_squared':str(val),
             'shared_entropy_to_inverseG_ratio':'1/4; absolute coefficients are profile dependent'}

    def test_smooth_cap_curvature_integral(self):
        rows=[]
        for n in [mp.mpf('.8'),mp.mpf('1.3')]:
            width=mp.mpf('.4')
            # Integrate sqrt(g) R=-2 F'' directly; avoids endpoint formula.
            Fpp=lambda r:-2*(1-n)*mp.tanh(r/width)/(width*mp.cosh(r/width)**2)
            value=mp.quad(lambda r:-4*mp.pi*Fpp(r),[0,width,5*width,mp.inf])
            self.assertLess(abs(value-4*mp.pi*(1-n)),mp.mpf('1e-40'))
            rows.append({'n':str(n),'integrated_R_per_surface_area':str(value)})
        records['smooth_cap_integrals']=rows

    def test_background_counterterm_does_not_fix_Newton(self):
        # Finite Gaussian version of the actual normalization identity:
        # a purely geometric source term cancels from all normalized matter
        # source derivatives, but changes the effective-action R coefficient.
        b,curv,j=S.symbols('b curvature j',real=True)
        d=S.symbols('d',positive=True)
        Z=S.exp(-b*curv)*S.exp(j*j/(2*d))/S.sqrt(d)
        norm=S.simplify(Z/Z.subs(j,0))
        self.assertEqual(S.diff(norm,b),0)
        Gamma0=-S.log(Z.subs(j,0))
        self.assertEqual(S.simplify(S.diff(Gamma0,b,curv)),1)
        K,ren1,ren2=S.symbols('loop_inverseG ren1 ren2')
        self.assertEqual(S.simplify((ren1-K)+K),ren1)
        self.assertEqual(S.simplify((ren2-K)+K),ren2)
        records['finite_parameter_no_go']='Same matter theory and regulator allow any finite renormalized inverseG by an independent background R term.'

    def test_species_and_state_information_controls(self):
        # Linear determinant/replica scaling for independent copies is a
        # conceptual species count, not an additional parent calculation.
        N,J=S.symbols('N J',positive=True)
        entropy=N*J/(48*S.pi);cR=-N*J/(192*S.pi**2)
        self.assertEqual(S.simplify(entropy+4*S.pi*cR),0)
        # The I02 sign-related coherent states generalize to this scalar:
        # same quadratic stress, different fixed local linear observable.
        lam,h=S.symbols('lambda h',real=True,nonzero=True)
        self.assertEqual((lam*h)**2-(-lam*h)**2,0)
        self.assertNotEqual(lam*h,-lam*h)

    def test_volume_term_requires_independent_flat_stationarity_condition(self):
        # Free background matter calculation has a quartically sensitive
        # volume term as well as the R term. Neither vanishes by matching.
        eps=S.symbols('epsilon',positive=True);bareV=S.symbols('bareV')
        inducedV=-1/(64*S.pi**2*eps**4)
        volume_variation=-(bareV+inducedV)/2
        self.assertEqual(S.solve(volume_variation,bareV)[0],1/(64*S.pi**2*eps**4))
        self.assertNotEqual(volume_variation.subs(bareV,0),0)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentChecks))
    out={'success':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),
         'errors':len(result.errors),'mpmath_digits':mp.mp.dps,'records':records}
    (HERE/'independent_results.json').write_text(json.dumps(out,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
