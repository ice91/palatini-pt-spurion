"""PTQ-SYM-FULL-CHAIN-AUDIT-01. Counterexamples are successful audit tests.
Pure analytic tests do not import the project. Provenance tests are identified explicitly.
Run with PYTHONPATH pointing only to audit working_copy, PYTHONDONTWRITEBYTECODE=1.
"""
import itertools
import json
from pathlib import Path
import sympy as s
import pytest

R=s.Rational
OUT=Path(__file__).resolve().parents[1]
g=[-1,1,1,1]
delta=lambda a,b:int(a==b)
def eps_up(*idx):
    if len(set(idx))<4:return 0
    return (-1)**sum(idx[i]>idx[j] for i in range(4) for j in range(i+1,4))
def eps_dn(*idx):return -eps_up(*idx)  # paper epsilon^0123=+1, Lorentz metric

def pure_trace(v):
    return s.MutableDenseNDimArray([R(1,3)*(g[a]*delta(a,c)*v[b]-g[a]*delta(a,b)*v[c]) for a,b,c in itertools.product(range(4),repeat=3)],(4,4,4))
def trace(T):return s.Matrix([sum(g[a]*T[a,b,a] for a in range(4)) for b in range(4)])
def axial(T):return s.Matrix([sum(eps_up(a,b,c,d)*T[b,c,d] for b,c,d in itertools.product(range(4),repeat=3)) for a in range(4)])
def norm(T):return s.expand(sum(g[a]*g[b]*g[c]*T[a,b,c]**2 for a,b,c in itertools.product(range(4),repeat=3)))
def paper_trace_projector(T):
    v=trace(T)
    return s.MutableDenseNDimArray([-x for x in pure_trace(v)],(4,4,4))
def paper_axial_projector(T):
    v=axial(T)
    return s.MutableDenseNDimArray([-R(1,6)*sum(eps_dn(a,b,c,d)*v[d] for d in range(4)) for a,b,c in itertools.product(range(4),repeat=3)],(4,4,4))
def scalar_curvature_constant(T):
    # D_mu V^a=partial_mu V^a+Gamma^a_{b mu}V^b; torsion = Gamma^a_{nu mu}-Gamma^a_{mu nu}
    K=lambda a,b,c:g[a]*R(1,2)*(-T[a,b,c]-T[b,c,a]+T[c,a,b])
    return s.simplify(sum(g[b]*sum(K(a,c,a)*K(c,b,b)-K(a,c,b)*K(c,b,a) for c in range(4)) for a,b in itertools.product(range(4),repeat=2)))

def test_projector_idempotence_and_selection():
    I=s.eye(2); U=s.diag(1,-1); P=(I+U)/2
    assert P*P==P and P*s.Matrix([1,0])==s.Matrix([1,0]) and P*s.Matrix([0,1])==s.zeros(2,1)
def test_A2_product_identity_counterexample():
    x=s.symbols('x'); P=lambda f:s.expand((f+f.subs(x,-x))/2)
    assert s.integrate(P(x)*P(x),(x,-1,1))==0
    assert s.integrate(P(x*x),(x,-1,1))==R(2,3)
def test_A9_hodge_conformal_counterexample():
    # Fixed p-form, g -> exp(2u)g: star scales exp((d-2p)u).
    d,p=4,2
    actual=d-2*p; paper=d-p  # 1/2 tr(h)*star -1/2 star(H.alpha), h_ab=2u*g_ab
    assert actual==0 and paper==2

def test_pure_trace_contraction_independent():
    v=list(s.symbols('v0:4')); T=pure_trace(v)
    assert trace(T)==s.Matrix(v) and axial(T)==s.zeros(4,1)
    assert s.expand(norm(T)-R(2,3)*sum(g[a]*v[a]**2 for a in range(4)))==0

def test_paper_A14_trace_projector_is_not_idempotent():
    T=pure_trace([1,2,3,4]); P=paper_trace_projector(T)
    assert trace(P)==-trace(T)
    assert paper_trace_projector(P)!=P

def test_paper_A15_axial_projector_sign():
    T=s.MutableDenseNDimArray.zeros(4,4,4)
    for b,c,d in itertools.product(range(4),repeat=3):T[b,c,d]=eps_dn(b,c,d,0)
    P=paper_axial_projector(T)
    assert axial(P)==-axial(T) and paper_axial_projector(P)!=P

def test_paper_A20_axial_norm_counterexample():
    T=s.MutableDenseNDimArray.zeros(4,4,4)
    for b,c,d in itertools.product(range(4),repeat=3):T[b,c,d]=eps_dn(b,c,d,0)
    S=axial(T); S2=sum(g[a]*S[a]**2 for a in range(4))
    assert norm(T)==-S2/6 and norm(T)!=S2/24

def test_C1_paper_trace_sign():
    v=s.Matrix([1,2,3,4]); eta=s.symbols('eta')
    # Eq16: T^a_bc=eta(delta^a_b v_c-delta^a_c v_b)
    traced=s.Matrix([sum(eta*(delta(a,b)*v[a]-delta(a,a)*v[b]) for a in range(4)) for b in range(4)])
    assert traced==-3*eta*v

def test_C1_IT_identity_survives_trace_sign():
    eta=s.symbols('eta'); v=list(s.symbols('v0:4')); T=pure_trace([-3*eta*x for x in v])
    assert s.expand(-norm(T)+6*eta**2*sum(g[a]*v[a]**2 for a in range(4)))==0

def test_single_vector_no_go():
    a,b=s.symbols('a b'); v=[1,2,3,4]
    T=s.MutableDenseNDimArray([a*(g[i]*delta(i,j)*v[k]-g[i]*delta(i,k)*v[j])+b*sum(eps_dn(i,j,k,l)*g[l]*v[l] for l in range(4)) for i,j,k in itertools.product(range(4),repeat=3)],(4,4,4))
    assert s.solve(list(trace(T))+list(axial(T)),[a,b])=={a:0,b:0}

def test_EH_contains_trace_quadratic_term():
    v=[s.Symbol('t'),0,0,0]; T=pure_trace(v); t2=-v[0]**2
    assert scalar_curvature_constant(T)==-R(2,3)*t2

def test_EH_axial_and_tensor_coefficients_nonzero():
    A=s.MutableDenseNDimArray.zeros(4,4,4)
    for i,j,k in itertools.product(range(4),repeat=3):A[i,j,k]=eps_dn(i,j,k,0)
    S=axial(A); S2=sum(g[i]*S[i]**2 for i in range(4))
    assert scalar_curvature_constant(A)==S2/24
    q=s.MutableDenseNDimArray.zeros(4,4,4)
    q[0,1,2]=1;q[0,2,1]=-1;q[1,2,0]=-1;q[1,0,2]=1
    assert trace(q)==s.zeros(4,1) and axial(q)==s.zeros(4,1)
    assert scalar_curvature_constant(q)==norm(q)/2

def test_alpha1_nonzero_does_not_guarantee_unique_irreps():
    alpha,M2=s.symbols('alpha M2',nonzero=True)
    # Full Eq13 in explicitly documented curvature convention, no new operators.
    axial_coeff=-alpha/6+M2/48
    assert s.simplify(axial_coeff.subs(alpha,M2/8))==0
    assert s.simplify((R(2,3)*alpha-M2/3).subs(alpha,M2/2))==0

def test_eta_formula_requires_full_action():
    t,v,a1,a3,M2=s.symbols('t v a1 a3 M2')
    L=(R(2,3)*a1-M2/3)*t*t+a3*t*v
    sol=s.solve(s.diff(L,t),t)[0]
    assert s.simplify(sol+3*a3*v/(4*(a1-M2/2)))==0
    assert s.simplify(sol+3*a3*v/(4*a1))!=0

def test_projective_scalar_integrability_obstruction():
    x,y=s.symbols('x y'); xi=[0,x]
    assert s.diff(3*xi[1],x)-s.diff(3*xi[0],y)==3  # d(d delta_eps)=0 impossible

def test_projective_shift_breaks_metric_compatibility():
    z=s.symbols('xi'); metric=s.diag(-1,1,1,1)
    shift=-z*metric-z*metric
    assert shift==-2*z*metric and shift!=s.zeros(4)

def test_ROD_determinant_second_order():
    x=s.symbols('x'); T=s.diag(R(3,4),-R(1,4),-R(1,4),-R(1,4))
    jac=s.sqrt((s.eye(4)+R(2,3)*x*T).det())
    assert s.series(jac-1,x,0,3).removeO()==-x*x/12

@pytest.mark.parametrize('sigma',[-1,1])
def test_CM_determinant_lemma_and_second_order(sigma):
    x=s.symbols('x'); metric=s.diag(-1,1,1,1); n=s.Matrix([1,0,0,0] if sigma==-1 else [0,1,0,0])
    ratio=s.factor((metric+R(2,3)*x*n*n.T).det()/metric.det())
    assert s.expand(ratio-(1+R(2,3)*x*sigma))==0
    assert s.series(s.sqrt(ratio)-1,x,0,3).removeO()==sigma*x/3-x*x/18

def test_ROD_CM_quadratic_mismatch():
    assert -R(1,12)!=-R(1,18)
    assert R(3,2)*R(1,12)==R(1,8) and R(3,2)*R(1,18)==R(1,12)

def test_pure_trace_Nieh_Yan_zero():
    # Forms represented by alternating coordinate components, constant coframe;
    # the repeated 1-form v also proves T^a wedge T_a=0 pointwise for variable v.
    v=[1,2,3,4]
    T=lambda a,b,c:delta(a,b)*v[c]-delta(a,c)*v[b]
    E=lambda a,b:delta(a,b)
    eT=[sum(g[a]*(E(a,i)*T(a,j,k)+E(a,j)*T(a,k,i)+E(a,k)*T(a,i,j)) for a in range(4)) for i,j,k in itertools.product(range(4),repeat=3)]
    TT=sum(eps_up(i,j,k,l)*g[a]*T(a,i,j)*T(a,k,l) for a,i,j,k,l in itertools.product(range(4),repeat=5))
    assert set(eT)=={0} and TT==0

def test_rank_nullity_published_system():
    a,b,c,d=s.symbols('a b c d'); M=s.Matrix([[a,b],[c,d]]); w=s.Matrix(s.symbols('w0:2'))
    assert M.adjugate()*M*w==M.det()*w
    # det!=0 => w=0; a nonzero 1D kernel requires rank exactly1 and det0.
    assert s.eye(2).nullspace()==[]

def test_config_intended_rational_rank1():
    M=s.Matrix([[R(3,4),-R(1,2)],[R(1,5),-R(2,15)]])
    assert M.det()==0 and M.rank()==1 and M.nullspace()==[s.Matrix([R(2,3),1])]

def test_config_literal_decimal_not_exact_rank1():
    M=s.Matrix([[R('0.75'),R('-0.50')],[R('0.20'),R('-0.13333333333333333')]])
    assert M.det()==R(1,400000000000000000) and M.rank()==2 and M.nullspace()==[]

def test_boundary_equivalent_routes_cannot_select_bulk_ratio():
    w1,w2,B,J1,J2=s.symbols('w1 w2 B J1 J2')
    # D is linear total divergence; quotient action maps J1,J2 to zero.
    quotient=lambda expr:s.expand(expr).subs({J1:0,J2:0})
    assert s.expand(quotient(w1*(B+J1)+w2*(B+J2))-(w1+w2)*B)==0

def test_total_divergence_coefficient_not_operator_identity():
    t=s.symbols('t'); f=t*(1-t); h=t**2*(1-t)**3
    assert s.integrate(s.diff(f,t),(t,0,1))==0
    assert s.integrate(s.diff(f,t)*s.diff(h,t)**2,(t,0,1))!=0

def test_A40_A38_flat_current_counterexample():
    t,x,A=s.symbols('t x A'); h=t**2+x**2
    div=s.diff(A*h*s.diff(h,t)/2,t)+s.diff(A*h*s.diff(h,x)/2,x)
    claimed=A*(s.diff(h,t)**2-s.diff(h,x)**2)/2
    assert s.simplify(div-claimed)!=0

def test_curvature_square_basis_dependency():
    riem,ric,curv=s.symbols('Riem2 Ric2 R2'); GB=riem-4*ric+curv; C2=riem-2*ric+curv/3
    assert s.expand(C2-(GB+2*ric-R(2,3)*curv))==0

def test_Ricci_squared_TT_offshell_nonzero():
    omega,k,H=s.symbols('omega k H'); box=(omega**2-k**2)
    # flat TT h11=H, h22=-H; Ric^(1)_ij=-1/2 Box hij
    RR=2*(-box*H/2)**2
    assert s.expand(RR-H**2*box**2/2)==0 and RR.subs(omega,0)!=0

def test_Weyl_squared_massless_branch_stays_luminal_flat():
    z,c=s.symbols('z c'); kernel=z+c*z*z
    assert kernel.subs(z,0)==0 and s.factor(kernel)==z*(c*z+1)

def test_NLO_derivative_count():
    derivatives_eps=0; derivatives_Sigma=2*(derivatives_eps+1)
    assert 2*(derivatives_Sigma+1)==6  # (nabla Sigma)^2 is not four derivatives

def test_tensor_speed_from_EH_ADM_principal_terms():
    # Direct ADM second variation, flat admissible patch, one TT plus mode.
    u,z=s.symbols('u z'); q=s.Function('q')(z); M2=s.symbols('M2',positive=True)
    gamma=s.diag(1+u*q,1-u*q,1); inv=gamma.inv()
    deriv=lambda expr,i:s.diff(expr,z) if i==2 else s.Integer(0)
    C=lambda i,j,k:s.simplify(sum(inv[i,l]*(deriv(gamma[l,k],j)+deriv(gamma[l,j],k)-deriv(gamma[j,k],l))/2 for l in range(3)))
    Ric=lambda i,j:sum(deriv(C(k,i,j),k)-deriv(C(k,i,k),j)+sum(C(k,k,l)*C(l,i,j)-C(k,j,l)*C(l,i,k) for l in range(3)) for k in range(3))
    R3=s.simplify(sum(inv[i,j]*Ric(i,j) for i,j in itertools.product(range(3),repeat=2)))
    spatial=s.expand(s.series(s.sqrt(gamma.det())*R3,u,0,3).removeO()).coeff(u,2)
    assert s.simplify(spatial-(2*q*s.diff(q,z,2)+R(3,2)*s.diff(q,z)**2))==0
    # q*q'' = (q*q')' - (q')^2, valid compact-support boundary improvement.
    spatial_bulk=s.expand(spatial).subs(q*s.diff(q,z,2),-s.diff(q,z)**2)
    assert s.simplify(spatial_bulk+s.diff(q,z)**2/2)==0
    qdot=s.symbols('qdot'); Kij=s.diag(u*qdot/2,-u*qdot/2,0)
    kinetic=s.sqrt(gamma.det())*(s.trace(inv*Kij*inv*Kij)-s.trace(inv*Kij)**2)
    kinetic2=s.series(kinetic,u,0,3).removeO().coeff(u,2)
    assert kinetic2==qdot**2/2
    K=s.diff(M2*kinetic2/2,qdot,2)
    G=-s.diff(M2*spatial_bulk/2,s.diff(q,z),2)
    assert K==M2/2 and s.simplify(G)==M2/2 and s.simplify(G/K)==1


def test_DoF_count_requires_constraint_classes():
    nconfig=6; nfirst=4
    assert (2*nconfig-2*nfirst)//2==2
    assert (2*nconfig)//2==6  # same kinetic-variable count is not constraint algebra

# Source/provenance probes: do not pretend these derive physics.
def test_existing_axial_code_wrong_for_spatial_axial_vector():
    import numpy as np
    from palatini_pt.palatini.torsion_decomp import axial_vector
    T=s.MutableDenseNDimArray.zeros(4,4,4)
    for a,b,c in itertools.product(range(4),repeat=3):T[a,b,c]=eps_dn(a,b,c,1)
    # Source uses epsilon_lower0123=+1, opposite to paper upper0123=+1.
    expected=-np.array(axial(T),dtype=float).ravel(); got=axial_vector(np.array(T.tolist(),dtype=float),np.diag(g))
    assert not np.allclose(got,expected)

def test_configured_coefficients_are_not_extracted(tmp_path):
    from palatini_pt.equivalence.coeff_extractor import coeff_vectors
    cfg={'coeffs':{}}
    for route,val in [('dbi',7),('closed',11),('cspp',13)]:
        p=tmp_path/(route+'.json');p.write_text(json.dumps({'I_T':val,'Seps':0}));cfg['coeffs'][route]=str(p)
    got=coeff_vectors(config=cfg,ibp_tol=1e-12)
    assert [got[k][0] for k in ['dbi','closed','cspp']]==[7,11,13]

def test_public_test_API_ignores_alpha():
    from palatini_pt.equivalence import dbi_order2
    assert dbi_order2(alpha=-100)==dbi_order2(alpha=100)

def test_strict_flag_does_not_block_toy_core(monkeypatch):
    import numpy as np
    monkeypatch.setenv('PALPT_REQUIRE_REAL_APIS','1')
    from palatini_pt.gw.quadratic_action import tensor_action_coeffs
    from palatini_pt.gw.tensor_mode import cT_of_k
    assert tensor_action_coeffs(99,locked=True)==(1.,1.)
    assert np.all(cT_of_k(np.array([1.,2.]),{'box_eps':1e9},lock=True)==1)

def test_strict_script_blocks_outer_fallback(tmp_path,monkeypatch):
    import scripts.fig_c2_coeff_compare as fig
    monkeypatch.setattr(fig,'REQUIRE_REAL',True)
    def missing(*args,**kwargs):raise RuntimeError('audit: absent physical API')
    monkeypatch.setattr(fig,'_residual_scan_via_module',missing)
    with pytest.raises(RuntimeError,match='absent physical API'):
        fig.run(config={'output':{'dir':str(tmp_path)}})

def test_degeneracy_figure_uses_synthetic_two_entry_alias():
    from scripts.fig_c3_degeneracy import _eigs_via_module
    got=_eigs_via_module({'spurion':{'seps_scale':0.2}})
    assert len(got)==2 and list(got)==[1.0016,1.002]

def test_FRW_scalar_TT_bilinear_vanishes():
    A,B,k,H=s.symbols('A B k H'); h=s.diag(H,-H,0); kv=s.Matrix([0,0,k])
    invariant_scalar_tensor=A*s.eye(3)+B*kv*kv.T
    assert s.trace(h)==0 and h*kv==s.zeros(3,1)
    assert s.trace(h*invariant_scalar_tensor)==0

def test_auxiliary_elimination_chain_rule_condition():
    x,a,b,c=s.symbols('x a b c',nonzero=True); T=s.symbols('T'); L=a*T*T+b*T*x+c*x*x
    Tstar=s.solve(s.diff(L,T),T)[0]
    assert s.simplify(s.diff(L.subs(T,Tstar),x)-s.diff(L,x).subs(T,Tstar))==0
    wrong=Tstar+x
    assert s.simplify(s.diff(L.subs(T,wrong),x)-s.diff(L,x).subs(T,wrong))!=0

def test_paper_NLO_config_not_read_by_core_model():
    import yaml
    from palatini_pt.gw.nlo import coeffs_from_config
    cfg=yaml.safe_load((OUT/'working_copy/configs/paper_grids.yaml').read_text())
    assert cfg['nlo']['coeffs']['Ricci_deps_deps_eff']==0.3
    assert coeffs_from_config(cfg)['Ricci_deps_deps_eff']==0.0

@pytest.mark.skip(reason='NOT_EXECUTABLE: no original ADM second variations or evaluated c0,cH,cY for four mixing entries')
def test_original_mixing_entries_from_ADM():pass
@pytest.mark.skip(reason='NOT_EXECUTABLE: nonzero PT/Hodge CS+ operator and normalization not specified')
def test_nonzero_CSplus_coefficient_from_operator():pass
@pytest.mark.skip(reason='NOT_EXECUTABLE: original K(w),G(w) physical Hessians absent; hardcoded functions excluded')
def test_original_locked_tensor_speed_from_action():pass
@pytest.mark.skip(reason='NOT_EXECUTABLE: physical Poisson/Dirac bracket algebra absent with fixed spurion')
def test_original_constraint_algebra():pass
