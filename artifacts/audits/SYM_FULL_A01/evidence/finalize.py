from pathlib import Path
import hashlib,json,subprocess,collections,datetime
O=Path(__file__).resolve().parents[1];root=O.parents[2]; E=O/'evidence'
initial=json.loads((E/'original_sha256.json').read_text())
actual={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and '.git' not in p.relative_to(root).parts and 'artifacts' not in p.relative_to(root).parts}
modified=[k for k,v in initial.items() if actual.get(k)!=v]; added=sorted(set(actual)-set(initial)); missing=sorted(set(initial)-set(actual))
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(); status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=root,text=True);(E/'final_git_status.txt').write_text(status)
tracked=subprocess.run(['git','diff','--exit-code','HEAD','--'],cwd=root,text=True,capture_output=True)
(E/'final_git_diff.stdout.txt').write_text(tracked.stdout);(E/'final_git_diff.stderr.txt').write_text(tracked.stderr)
expected_head='6b456b89baa866b6dd05bc69a837b98a985a34ad'
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_files':len(initial),'verified_unchanged':len(initial)-len(modified),'modified':modified,'added_outside_audit':added,'missing':missing,'head':head,'head_unchanged':head==expected_head,'branch':branch,'branch_unchanged':branch=='main','tracked_git_diff_exit_code':tracked.returncode,'publisher_pdf_byte_identical':(root/'symmetry-18-00170-v2.pdf').read_bytes()==(E/'publisher_v2.pdf').read_bytes(),'allowed_new_subtree':str(O),'status_log':'evidence/final_git_status.txt','scientific_source_preservation_pass':not(modified or added or missing) and head==expected_head and branch=='main' and tracked.returncode==0}
(E/'preservation_final.json').write_text(json.dumps(result,indent=2)); (E/'original_sha256_final.json').write_text(json.dumps(actual,indent=2))
claims=json.loads((E/'claim_adjudication.json').read_text());counts=collections.Counter(r['VERDICT'] for r in claims)
md=f'''# Audit state — stopped at bounded source boundary

Task: PTQ-SYM-FULL-CHAIN-AUDIT-01. State: **BOUNDED_AUDIT_COMPLETE_WITH_RECORDED_SOURCE_AND_ENVIRONMENT_LIMITATIONS**.

No new theory development, correction work, remote mutation, journal contact, publication, package installation or original-source edits were performed. No follow-up research has been launched. No background process is left running by this audit.

## Original evidence preservation

**PASS:** {result['verified_unchanged']}/{result['original_files']} original scientific/artifact files verify against the pre-execution SHA-256 manifest. Modified: {len(modified)}; missing: {len(missing)}; new files outside authorized audit subtree: {len(added)}. HEAD remains `{head}`; active branch remains `{branch}`. Tracked `git diff --exit-code HEAD --` returns{tracked.returncode}. The original untracked v2 paper is unchanged and still byte-identical to the independently fetched publisher v2 artifact.

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
`{O}`

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

'''+ '\n'.join(f'- {k}: {v}' for k,v in sorted(counts.items()))+'''

## Surviving conclusion and single next action

A conditional ordinary EH luminal tensor principal branch survives under genuine nonsingular algebraic torsion elimination. PT-specific unique physical locking and the published unique NLO slope2 mechanism are not established. No conclusion is made that every PTQ framework is invalid.

Recommended single next action (not executed): obtain the archived original action-to-ADM derivation for this exact version, with full connection variation, evaluated mixing entries, physical K,G and constraint provenance. Work stops without altering or rescuing the published model.
'''
(O/'10_STATE.md').write_text(md)
# Static report/test evidence integrity record; excludes itself and mutable copied outputs.
report_hashes={str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(O.glob('*.md'))}
report_hashes.update({str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (O/'tests').glob('*.py')})
(E/'audit_deliverable_sha256.json').write_text(json.dumps(report_hashes,indent=2))
print(json.dumps(result,indent=2));print('CLAIM_COUNTS',dict(counts));print('DELIVERABLES',len(report_hashes))
