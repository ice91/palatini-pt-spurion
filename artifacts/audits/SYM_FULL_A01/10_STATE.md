# Audit state — stopped at bounded source boundary

Task: PTQ-SYM-FULL-CHAIN-AUDIT-01. State: **BOUNDED_AUDIT_COMPLETE_WITH_RECORDED_SOURCE_AND_ENVIRONMENT_LIMITATIONS**.

No new theory development, correction work, remote mutation, journal contact, publication, package installation or original-source edits were performed. No follow-up research has been launched. No background process is left running by this audit.

## Original evidence preservation

**PASS:** 131/131 original scientific/artifact files verify against the pre-execution SHA-256 manifest. Modified: 0; missing: 0; new files outside authorized audit subtree: 0. HEAD remains `6b456b89baa866b6dd05bc69a837b98a985a34ad`; active branch remains `main`. Tracked `git diff --exit-code HEAD --` returns0. The original untracked v2 paper is unchanged and still byte-identical to the independently fetched publisher v2 artifact.

Hash manifests: `evidence/original_sha256.json` and `evidence/original_sha256_final.json`. Final machine-readable verification: `evidence/preservation_final.json`. Initial/final Git observations: `evidence/initial_git_state.txt`, `evidence/final_git_status.txt`. New audit files naturally appear as untracked additions inside the authorized subtree; this is not a change to the pre-existing evidence.

## Transient cache correction

The first independent-test development invocations created a new default `.pytest_cache` at the original root despite putting pytest basetemp inside the audit directory. Final verification detected these five added cache files; none existed in the initial manifest and none was an original source file. They were moved intact to `evidence/relocated_root_pytest_cache/`. The final independent runner executes from working_copy. No new file remains outside the authorized subtree. This transient output-location mistake and the failed pre-relocation verification are retained in `evidence/cache_relocation.json` and `preservation_pre_cache_relocation.json`; they are not concealed as an initially clean run.

## Execution outcomes

|Run|Outcome|
|---|---|
|Original44-case suite|42 passed,2 skipped (missing installed palpt entrypoint)|
|Literal make paper-test|exit2: pytest command not on PATH|
|Equivalent module-based strict paper suite|42 passed,2 skipped|
|make mixing|exit0: configured ratios consistent|
|make paper|exit2 after eight modules: unsupported Python3.9 union annotation in final flux module|
|Independent46-case suite|42 passed,4 NOT_EXECUTABLE, exit0|
|Original artifact sidecars|20/20 match|
|Original source/artifact hash preservation|131/131 unchanged|

**MISSING_ENVIRONMENT:** full project-supported Python>=3.10 runtime with required dependencies was not available in an existing executable environment. No installation was authorized or attempted. Exact diagnostic runtime and package inventory are recorded in00/evidence. Mathematical counterexamples are exact symbolic checks independent of the failed annotation path.

## Missing-source stop conditions

- No evaluated original ADM c0,cH,cY table / four mixing entries.
- No explicit nonzero CS+ operator/normalization beyond ordinary NY identity.
- No original physical constraint-reduced weighted K(w),G(w) Hessians.
- No original fixed-spurion physical Poisson/Dirac constraint algebra.

These four checks are NOT_EXECUTABLE rather than filled with invented physical inputs. Their absence is a completed audit finding, not a claim that the missing theorems are thereby true or false. Demonstrations and explicit contradictions are separately adjudicated.

## Deliverable inventory

All eleven requested reports exist under:
`/Users/rocky/Downloads/palatini-pt-spurion/artifacts/audits/SYM_FULL_A01`

- `00_SOURCE_MANIFEST.md`
- `01_SOURCE_AND_CLAIM_MAP.md`
- `02_BASELINE_REPRODUCTION.md`
- `03_C1_INDEPENDENT_AUDIT.md`
- `04_C2_THREE_ROUTE_AUDIT.md`
- `05_C3_TENSOR_AUDIT.md`
- `06_NLO_AND_FIGURES.md`
- `07_ADVERSARIAL_TESTS.md`
- `08_ROOT_CAUSE_ANALYSIS.md`
- `09_FINAL_ADJUDICATION.md`
- `10_STATE.md`

Additional evidence: `tests/test_independent_audit.py`, `evidence/claim_adjudication.json` (33 complete claim records), command/stdout/stderr logs, runtime call graphs, PDF identity/text/page renders, artifact hashes/comparisons, line-numbered source snapshots and an isolated `working_copy/`. All code-generated output/caches/temporary tests remain inside this subtree.

## Adjudication counts

Counts summarize records of different scopes (including valid narrow lemmas and separate false broad claims); they are not a numerical score for the whole theory.

- BLOCKED_BY_MISSING_SOURCE: 2
- CONDITIONAL: 7
- CONTRADICTED_AS_WRITTEN: 14
- DEMONSTRATION_ONLY: 2
- PROVED_WITHIN_STATED_DOMAIN: 5
- UNVERIFIED: 3

## Surviving conclusion and single next action

A conditional ordinary EH luminal tensor principal branch survives under genuine nonsingular algebraic torsion elimination. PT-specific unique physical locking and the published unique NLO slope2 mechanism are not established. No conclusion is made that every PTQ framework is invalid.

Recommended single next action (not executed): obtain the archived original action-to-ADM derivation for this exact version, with full connection variation, evaluated mixing entries, physical K,G and constraint provenance. Work stops without altering or rescuing the published model.
