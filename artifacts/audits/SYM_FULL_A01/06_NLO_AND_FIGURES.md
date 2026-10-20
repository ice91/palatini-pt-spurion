# NLO and figure validity

## Verdict

The program reproducibly implements a chosen k² offset, but **the published complete four-derivative basis and unique Weyl-squared source of a slope2 massless tensor-speed deviation are not established; several operator statements are contradicted as written**. This is stronger than a missing numerical fit and narrower than a rejection of every possible higher-derivative correction.

Sources: PDF p.9 Table2, §§8.2–8.5 pp.21–23, App.H pp.46–49 A72–A93; `gw/nlo.py`, `scripts/fig_nlo_offsets.py`, `tests/test_nlo.py`, notebook30 and paper_grids.yaml.

## Basis completeness and internal dependencies

In4D Levi–Civita geometry the pointwise curvature identities are

```
GB=Riemann²−4Ricci²+R²,
C²=Riemann²−2Ricci²+R²/3
  =GB+2Ricci²−(2/3)R².
```

Therefore the listed GB,R²,Ricci²,C² are not independent. On a domain where GB is a valid boundary, a classification that makes R² and Ricci² entirely TT-neutral while C² supplies a distinct nonzero TT principal direction is inconsistent unless different field-redefinition/on-shell conventions are explicitly applied and tracked. The same equivalence convention must be applied to all rows. This exact algebraic dependency is independently tested.

With epsilon derivative counting fixed by §3, Sigma=(d epsilon)² has two derivatives; `(nabla Sigma)²` has **six**, not four. Although homogeneous fixed-gradient backgrounds can make some such terms vanish, that does not make the operator itself four-derivative or prove a complete basis. Scalar projection and projective consistency are also obstructed by the geometric issues in report03. The repository's two ASCII labels do not enumerate geometric scalar contractions or establish completeness.

No new operators are proposed here. Completeness is not rescued by expanding the model or changing the truncation.

## Independent TT check of curvature squares

On a flat admissible patch, a transverse-traceless h has

```
R^(1)=0,
R^(1)_mu_nu=−(1/2)Box h_mu_nu.
```

Thus `[(sqrt(-g) Ricci²)]^(2)=(1/4)(Box h_ij)²`, nonzero **off shell**. With h11=H,h22=−H and Fourier frequency omega,wavenumber k, this is `(H²/2)(omega²−k²)²`. This is an explicit polynomial counterexample to Table2's unconditional “no TT contribution” statement for Ricci². The fact that it vanishes on a leading-order Einstein wave is an on-shell/order-reduction statement, not an off-shell density identity.

For R², R^(1)=0 on flat TT does remove this leading square, but on curved backgrounds `2Rbar R^(2)` and background/constraint terms require tracking; “no TT” and “equal renormalization” are different statements. Appendix H itself shifts from the Table2 no-contribution wording to a background-dependent lockstep renormalization claim. Such claims can hold on narrower backgrounds with specified equations and field redefinitions; the general smooth-background statement is not shown.

By the curvature identity, C² modulo GB on the same flat TT patch yields a multiple of `(Box h)²`, not an isolated `(spatial Laplacian h)²` term with the time-derivative partners absent. An EH-plus-curvature-square kernel on this branch has the schematic polynomial

```
z(1+c z)=0,  z=omega²−k².
```

The ordinary massless root remains z=0. The other higher-derivative root must be treated as an extra branch or excluded by a stated low-energy order-reduction prescription. One cannot retain only k⁴ from `(omega²−k²)²` and call the resulting shift an inevitable massless slope2 prediction. On a curved or preferred-background system further terms might change propagation, but the source must actually derive them. The flat local calculation refutes an unconditional Weyl²→nonzero b massless-branch claim and the off-shell reduction used to motivate it; it does not assert a theorem for all modified-gravity models.

In the reduced spatial four-derivative action displayed in A89/A92, the sign of the induced dispersion also depends on the coefficient/sign convention for that term. The source gives no exact Wilson-coefficient map b(c_C²,background) with kinetic normalization and branch choice. Its `b proportional c_C²` order estimate is not that calculation.

## Source versus claimed physical operator

`gw/nlo.py` explicitly calls itself a small parameterization. It chooses

```
a=gradT_sq_eff/Lambda²,
b=Ricci_deps_deps_eff/Lambda²,
DeltaK=a k²,  DeltaG=(a+b)k²,
reported_delta_cT²=DeltaG−DeltaK=b k².
```

It does not vary either named density. The paper instead names Weyl² as the unique dangerous operator. No code transforms the two effective label coefficients into c_C², integrates torsion at NLO, or reduces an actual four-derivative TT action. The ASCII-basis test checks strings only.

Moreover for K=G=1 at LO, the **exact ratio** in this assigned model is

```
(1+DeltaG)/(1+DeltaK)−1 = b k²/(1+a k²),
```

not simply b k² unless truncated to first order in k²/Lambda². `predict_offsets` is an asymptotic difference, not an exact high-frequency speed function. Slope2 is mathematically expected only where the leading nonzero coefficient is present, scales are consistent and higher orders are negligible. For b=0 there is no nonzero log–log slope to predict.

The figure script separately calculates `b*k²/Lambda²` directly from `nlo.coeffs.Ricci_deps_deps_eff`. It does not call `delta_KG` or a physical operator reduction. The script reads nested `nlo.coeffs`, while the core API reads flat `nlo`; passing paper_grids.yaml unchanged to the core API yields its zero defaults. Notebook30 uses a flat mapping, so its chosen curve is similar by construction rather than a common verified physical coefficient path.

`tests/test_nlo.py` checks a nonempty ASCII list, a zero-parameter limit and doubling k→fourfold offset. That validates the selected formula, not its operator origin. The plotted fit recovers slope2 from samples whose exponent was assigned2. In the optional annotation path `CFG_CACHE` is never populated, so a Lambda estimate is not reliably sourced from the input even though slope fitting occurs. This is a secondary software issue.

## Dimensional qualifications

For dimensionless epsilon, Sigma R has four derivative powers and dimensionmass⁴, whereas `(nabla Sigma)²` has dimensionmass⁶. Pure curvature squares also have dimensionmass⁴. Writing all of them with a common1/Lambda² and dimensionless Wilson coefficients requires an explicit overall normalization and a consistent dimension basis. Eq15/A75 as printed omits that clarification; App.H later introduces M_T²/8 in the reduced tensor action. The audit records the normalization gap rather than supplying missing scales. This gap does not change the independently verified curvature identity or off-shell counterexample.

## Figure classification

|Paper / repository figure|Actual source|Classification|
|---|---|---|
|PDF Fig2 / repo fig1 C1 components|Construct pure trace then decompose|Consistency check of imposed ansatz; no dynamics|
|PDF Fig4 / repo fig2 alignment|Array of zeros|Illustration of assigned alignment|
|PDF Fig5 / repo fig3 C2 residual|Compare equal supplied coefficients|L4 consistency plot|
|PDF Fig6 / repo fig9 flux|1+c R^(−sigma)|Illustrative convergence curve; final module not regenerated here|
|PDF Fig7 / repo fig4 heatmap|1+.02(X+Y)²|Illustrative effective parameter scan, mismatched weight interpretation|
|PDF Fig8 / repo fig5 dispersion|ones for locked; tanh² for unlocked|Illustrative assigned dispersion|
|PDF Fig9 / repo fig7 waveform|Two identical sinusoids|Consistency illustration, no physical wave evolution|
|PDF Fig10 / repo fig6 DoF|Synthetic two-entry eigenvalue array|Illustration; no physical constraint count|
|PDF Fig11 / repo fig8 NLO|Chosen b*k²/Lambda² and fitted log slope|Parameterized forecast demonstrator, no derived unique prediction|

All20 original adjacent MD5 sidecars match; that establishes integrity only. Eight figure modules executed before the Python compatibility error in the final flux module. No matching/reproduced image is promoted to evidence for a physical theorem. The figures provide no observational data validation or independent measured frequency-band scaling.
