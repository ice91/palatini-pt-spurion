"""Secondary check: SAME I01 detector/worldline/switch; change field preparation only.
Compute finite one-particle excess, not a new equilibrium rate.
"""
from pathlib import Path
import json
import mpmath as mp
HERE=Path(__file__).resolve().parent
mp.mp.dps=45
def rate(nu):
    return nu/mp.expm1(2*mp.pi*nu) if nu else 1/(2*mp.pi)
def g(t):
    # a=ell=1; derived from F=1/[sqrt(pi)*(1-i*exp(-tau))].
    return 1/(2*mp.sqrt(mp.pi)*(1+mp.j*mp.sinh(t)))
def amplitude(e,T):
    fun=lambda t:mp.exp(-t*t/(2*T*T))*mp.exp(-mp.j*e*t)*g(t)
    return mp.quad(fun,[-mp.inf,-8,-2,0,2,8,mp.inf])
rows=[]
E=mp.mpf('0.5')
for T in [mp.mpf(2),mp.mpf(5)]:
    Ip=amplitude(E,T);Im=amplitude(-E,T)
    excess=abs(Ip)**2+abs(Im)**2
    vacuum=T*mp.quad(lambda y:mp.exp(-y*y)*rate(E+y/T),[-mp.inf,-T*E,0,mp.inf])
    assert excess>0 and vacuum>0
    rows.append({'a':'1','ell':'1','E':'1/2','T':str(T),'I_E_real':str(Ip.real),'I_E_imag':str(Ip.imag),
                 'I_minusE_real':str(Im.real),'I_minusE_imag':str(Im.imag),
                 'vacuum_response':str(vacuum),'state_excess':str(excess),
                 'one_particle_response':str(vacuum+excess),
                 'scope':'leading response F; multiply by lambda_detector^2 |m_eg|^2 for probability'})
# The exact no-switch excess is finite while vacuum total response is infinite;
# this is a limit of the EXCESS only, not a total probability claim.
def amp_inf(e):
    return mp.sqrt(mp.pi)*e*mp.exp(-mp.pi*e/2)/mp.sinh(mp.pi*e) if e else 1/mp.sqrt(mp.pi)
I=mp.quad(lambda t:mp.exp(-mp.j*E*t)*g(t),[-mp.inf,-8,-2,0,2,8,mp.inf])
assert abs(I-amp_inf(E))<mp.mpf('1e-35')
result={'all_pass':True,'rows':rows,'unwindowed_excess_at_E_half':str(abs(amp_inf(E))**2+abs(amp_inf(-E))**2),
        'unwindowed_amplitude_error':str(abs(I-amp_inf(E))),
        'identity':'DeltaF_chi(E)=|integral chi*exp(-iE tau)*Fdot|^2 + |integral chi*exp(+iE tau)*Fdot|^2',
        'commutator_check':'DeltaF(E)=DeltaF(-E); exact from the amplitude decomposition',
        'not_inferred':'no reconstruction of global correlations from identical accessible restrictions'}
(HERE/'detector_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
