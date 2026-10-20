# Predictivity Boundaries in Free-Function Geometric Gravity:
# Constitutive Freedom, Functional Identifiability,
# and a Worked Gravitational Example

**Document type:** technical whitepaper (not a journal submission)  
**Audience:** modified-gravity, mathematical physics, inverse-problem,
and scalar-tensor / EFT readers  

This note records a **worked example**. It does not propose a new
abstract inverse-problem theorem, does not claim observational
exclusion, and does not assert that no measurement in nature could
ever identify the constitutive maps of the model.

---

## Abstract

A geometric gravitational model whose remaining constitutive content
consists of two functions of one scalar is not constitutively closed
by a single regular algebraic relation, nor by local well-posedness of
the metric problem at a fixed constitutive package. In a linearized
proper-time (clock) response family for that model, the constitutive
linearization has an explicit nontrivial functional kernel: variations
of the maps off the image of the background scalar, and a
\(Z\)-direction that is linearly invisible on a constant-scalar
background. Sensitivity of one effective combination is therefore not
functional identifiability. Re-evaluating the same operator on another
state, or specializing the same field equation to a homogeneous
cosmology, need not raise constitutive rank. Finite-dimensional
parameterizations can appear to fit data while a functional kernel
remains. Constitutive measurement, if ever available, must reconstruct
the maps from an independent interface and then predict on a genuinely
independent target; same-catalog reconstruction is not prediction.
The example does not show that every modified-gravity theory is
non-identifiable, nor that every conceivable observable in this model
fails.

---

## 1. Introduction

Many effective gravitational models retain **free functions** of a
scalar field rather than a finite list of couplings. Reconstruction
programs in scalar-tensor theory, cosmological inverse problems, and
EFT coefficient functions all face the same structural question: which
observations determine those functions, and which only determine
combinations, moments, or gauge-equivalent representatives?

This whitepaper isolates that question in a PT-even geometric setting
whose residual constitutive data, after a stated equivalence, are two
maps of one scalar. The operational family used as a worked
observation operator is a linearized proper-time response along
declared timelike worldlines. The concrete result is a **named
kernel** for that family, together with methodological cautions that
are standard in inverse problems once the kernel is exhibited.

What follows is not a survey of modified gravity, not a data analysis,
and not a claim that the model is observationally ruled out.

---

## 2. Free Functions Versus Free Parameters

Two real parameters are a finite chart. Two functions of a scalar
argument are an infinite-dimensional unknown, even after a
finite-dimensional gauge or rescaling quotient.

A regular pointwise algebraic relation \(C(f,Z)=0\) typically solves
for one map in terms of the other and leaves a free function. Writing
\(Z=F(f)\) with unspecified \(F\) does not reduce the unknown: it
renames it. Reduction to finitely many constants requires an ODE/jet
problem in the scalar argument whose free data are finite, or an
injective measurement of both maps. A single global number cannot
replace a map.

These remarks are elementary. They are recorded only to block the
slogan that “two functions are two parameters.”

---

## 3. Law, State, and Constitutive Data

For a metric field theory it is useful to separate:

- **Law:** the tensorial form of the field equation.
- **State:** Cauchy data, sources, and realized metric solutions at a
  *fixed* constitutive package.
- **Constitutive data:** coefficient functions that are prescribed
  rather than varied, and that are not evolved by their own well-posed
  initial-value problem.
- **Environment:** probes and preparations.
- **Gauge:** equivalences already quotiented.

Ordinary multiplicity of Einstein-like solutions at fixed
constitutive data is not constitutive underdetermination. Residual
**maps** in the coefficient slot are. Promoting those maps to
dynamical fields does not automatically close the theory: it typically
relocates the freedom into new initial and boundary data.

---

## 4. Two-Function Geometric Model

The worked model is a PT-even geometric gravitational theory in which,
after reparameterization of a scalar argument, bulk-null integrations
by parts, and a joint rescaling with an overall coupling, the remaining
constitutive unknown is a class

\[
c
=
\bigl[(f_1(\phi),\, Z(\phi))\bigr]_{\sim}.
\]

An auxiliary potential is not treated here as an independent
vacuum-response modulus. The pair is evaluated on a prescribed
background scalar \(\phi_{\mathrm{bg}}\); coefficients in the
linearized metric operator are compositions with that image.
Off-image jets of \((f_1,Z)\) do not enter the linearized problem at
that background.

The model is used as an example of free-function geometric gravity. No
claim is made that it is the unique such theory, or that it is
observationally preferred.

---

## 5. Observation Operator

The named observation family is an indexed linearized proper-time
response. For a declared timelike worldline with trigger events, one
forms a fractional duration shift relative to the background metric
duration. A source increment \(J\) produces a linearized metric
perturbation, hence a linearized clock increment. The constitutive
unknown \(c\) enters through the linearized gravitational operator and
its forcing coefficients.

Schematically,

\[
\delta c
\;\longrightarrow\;
(\delta L,\,\delta F)
\;\longrightarrow\;
\delta U
\;\longrightarrow\;
\delta h
\;\longrightarrow\;
\delta Y,
\]

where \(Y\) denotes the family of clock increments. Distinct worldline
contractions can emphasize different solution sectors of the
linearized metric problem (tensor versus scalar-type pieces). Channel
separation inside the **same** family is not automatically injectivity
on the two-function space.

No lensing map, gravitational-wave amplitude map, or galactic
rotation-curve dictionary is used in this note. Their absence is a
scope limitation, not a proof that those observables could not, in
some other analysis, carry constitutive information.

---

## 6. Functional Kernel

Let \(DM_c\) be the linearized map from constitutive variations
\((\delta f_1,\delta Z)\), modulo the stated equivalence, to
variations of the clock family.

The following directions lie in the kernel of that map for the named
family (linearized, weak-field setting):

1. **Off-image directions.** If \(\delta c\) is supported off
   \(\mathrm{im}(\phi_{\mathrm{bg}})\), it does not enter the
   linearized operator at that background, hence produces \(\delta Y=0\).
2. **Constant-scalar \(Z\)-direction.** On a constant-\(\phi\)
   background, the linear response sees an effective combination
   involving \(f_1\) at that value (an overall coupling over \(f_1\))
   and is linearly insensitive to \(Z\).

Infinitesimal reparameterization of \(\phi\) is already quotiented and
is not counted as a new kernel direction. A generic amplitude
variation of \(f_1\) at frozen overall coupling is **not** in the
kernel on a nondegenerate probe.

The kernel is background-dependent and is an output-relative nullspace
of the named family, not a statement of full theory equivalence.

Consequently, the two maps are **not** recoverable as functions from
this family. Full reconstruction is not obtained. Local
response-identifiability of the constitutive class is false for this
operator.

---

## 7. Sensitivity Versus Identifiability

A combination may be visible—for example an effective amplitude on a
constant-scalar background—without the map \(DM_c\) being injective.
Rank of a single real clock increment is at most one, which cannot
separate two function-directions. A continuously indexed family of
the same kernel can increase channel separation and still leave a
functional nullspace.

Identifiability is a property of the kernel (and, globally, of fibres
of the nonlinear map). It is not implied by nonzero sensitivity, by
the number of datapoints, or by a well-conditioned discrete Jacobian
in a truncated basis.

This note does **not** prove nonlinear global uniqueness failure
beyond the exhibited linear kernel and the elementary observation that
distinct maps agreeing on \(\mathrm{im}(\phi_{\mathrm{bg}})\) (and,
when relevant, agreeing in the visible combination at constant
\(\phi\)) produce the same linearized clocks.

---

## 8. Why Multiple Regimes Need Not Add Rank

Additional equations raise constitutive rank only if they are
independent constraints on the maps \((f_1,Z)\).

The following do **not**, by themselves, supply that independence:

- specializing the same tensor field equation to a homogeneous
  isotropic ansatz (a state reduction, not a new law);
- differential identities (Bianchi / Noether type) of the same
  equation, which constrain backgrounds rather than determining both
  constitutive maps;
- gauge equivalences;
- re-evaluating the same observation operator on another source or
  another background (further samples of one kernel);
- characteristic cones that are insensitive to the maps as functions;
- global numbers that are not functions of the scalar argument.

In the worked model, audited additional sectors do not reduce the
two-function remainder to a finite chart. This is a case analysis, not
a theorem that no field theory can ever gain rank by combining
regimes.

---

## 9. Finite-Basis Reconstruction Caution

Projecting \((f_1,Z)\) onto a finite polynomial, spline, or network
basis can make a discrete inverse appear square or even well-posed.
If a functional kernel exists, that appearance is a property of the
truncation. Modes in the kernel that are invisible to the observation
operator remain undetermined at every finite dimension; as the basis
is refined, those directions reappear as ill-posed or null components.

No separate numerical experiment is reported here. The caution follows
from the kernel in Section 6. A finite-basis fit is not a proof of
functional identifiability.

---

## 10. Constitutive Measurement Versus Target Fitting

A scientifically legitimate constitutive program would reconstruct
\(c\) from an independent measurement interface, freeze it, and
predict a genuinely independent target, testing **shape**, not only a
scale. Using the same observational catalog both to determine \(c\)
and to confirm the prediction is target fitting, not constitutive
measurement.

Universality of \(c\) across independently prepared systems is a
hypothesis, not a theorem of the model. Disagreement of independently
reconstructed classes, when reconstruction exists, would falsify that
hypothesis; averaging incompatible maps into one “effective” function
would not restore it.

This note contains no catalog analysis.

---

## 11. Worked PT-Even Geometric Example

Collecting the example:

- Unknown: two constitutive maps of one scalar, after the stated
  equivalence.
- Operator: linearized indexed proper-time response.
- Kernel: off-image variations; constant-scalar \(Z\)-null direction.
- Visible: an effective combination on the constant-scalar sector for
  nondegenerate probes; further channel separation on nonconstant
  backgrounds, without a proved two-function injectivity.
- Not used: lensing, gravitational-wave amplitudes, rotation curves,
  cosmological parameter fits.

The example shows how a geometrically natural clock family can be
sensitive and still fail to identify free constitutive functions. It
does not show that clocks are useless, nor that the model is empty of
structure: at a *fixed* constitutive package, the linearized response
remains a definite (indexed) prediction.

---

## 12. Limits of the Present Result

- Linearized / weak-field constitutive sensitivity; not a complete
  nonlinear inverse.
- Named operator only; other observables are out of scope.
- No claim that no conceivable interface can identify the maps.
- No observational exclusion and no cosmological closure.
- No uniqueness statement for a physical “branch” of solutions.
- Multi-regime rank is a case framework, not an abstract lemma for
  all theories.
- Truncation caution is not a standalone numerical theorem.

---

## 13. Conditions for Reopening Predictive Closure

Predictive closure of the constitutive slot would require at least one
of the following, stated as scientific conditions rather than project
rules:

1. A uniquely determined generating principle for \((f_1,Z)\) whose
   own arbitrary content is smaller than two maps, with a nontrivial
   solution sector.
2. An independent field-theoretic constraint on the pair that is not a
   specialization, identity, or gauge relation of the existing
   equation.
3. An operational family with independent physical provenance, a
   constitutive kernel distinct from the clocks, coverage of the
   scalar argument as an independently known interval, and access to
   the \(Z\) sector, without a matching increase in nuisance functions.
4. A genuine bridge to another theory sector that supplies constitutive
   rank, not a resemblance.
5. Empirical information that reconstructs the maps without using the
   later confirmation target as the reconstruction data.

Until one of these exists, the constitutive pair remains a
two-function remainder relative to the named clocks.

---

## 14. Discussion

Free-function modified gravity often looks predictive once a
convenient interpolation or a finite basis is chosen. The present
example is a reminder that **choice of a function is not measurement
of a function**, and that **visibility of a combination is not
injectivity**. Scalar-tensor reconstruction and cosmological inverse
problems already know related degeneracies; the contribution here is a
concrete kernel in a clock-response setting, not a new general theory
of inverse problems.

A skeptical reading—“this is tautological once a kernel is
exhibited”—is accepted for the abstract implication. The nontrivial
work in the example is the **construction** of the kernel for the
named family, including the constant-scalar \(Z\)-null direction and
the off-image directions.

---

## 15. Conclusion

In the worked PT-even geometric model, two constitutive functions of
one scalar remain after the stated equivalences. The linearized
proper-time response family has a nontrivial functional kernel and
does not reconstruct the pair. Sensitivity, extra samples of the same
operator, specializations of the same law, and finite-basis fits do
not remove that remainder. The result is a predictivity **boundary**
for this observation architecture, not a verdict on the model’s
existence or on modified gravity as a whole.
