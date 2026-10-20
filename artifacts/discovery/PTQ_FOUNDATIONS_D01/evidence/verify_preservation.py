"""Read-only comparison of originals; writes records only in this discovery evidence dir."""
from pathlib import Path
import hashlib,json,subprocess,datetime
HERE=Path(__file__).resolve().parent
OUT=HERE.parent
ROOT=OUT.parents[2]
initial=json.loads((HERE/'initial_state.json').read_text())
assert str(ROOT)==initial['pwd']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base=initial['sha256']
changed=[p for p,h in base.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
external=[p for p,h in initial['external_sha256'].items() if not Path(p).is_file() or sha(Path(p))!=h]
prefix=str(OUT.relative_to(ROOT))+'/'
extra=[str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts and str(p.relative_to(ROOT)) not in base and not str(p.relative_to(ROOT)).startswith(prefix)]
state={k:subprocess.check_output(args,text=True,cwd=str(ROOT)) for k,args in {'HEAD':['git','rev-parse','HEAD'],'branch':['git','branch','--show-current'],'status':['git','status','--short'],'refs':['git','show-ref']}.items()}
diff=subprocess.run(['git','diff','--exit-code'],cwd=str(ROOT),capture_output=True).returncode
staged=subprocess.run(['git','diff','--cached','--exit-code'],cwd=str(ROOT),capture_output=True).returncode
assert not changed and not external and not extra
assert all(state[k]==initial['git'][k] for k in state)
assert diff==staged==0
expected=['00_RESEARCH_CHARTER.md','01_AXIOMS_AND_CANDIDATE_MODELS.md','02_PT_GEOMETRIC_CONSTRUCTION.md','03_OPERATOR_RESTRICTION_DERIVATION.md','04_ACTION_VARIATION_AND_SCHUR.md','05_PHYSICAL_SYMBOL_AND_COUNTEREXAMPLES.md','06_CONTROL_MODEL_AND_NOVELTY.md','07_RESEARCH_SYNTHESIS.md']
assert sorted(p.name for p in OUT.glob('*.md'))==expected
assert all(p.is_dir() and p.name=='evidence' or p.is_file() and p.name in expected for p in OUT.iterdir())
res=json.loads((HERE/'derivation_results.json').read_text())
tests=json.loads((HERE/'independent_test_results.json').read_text())
ec=json.loads((HERE/'EC_connection_results.json').read_text())
assert res['all_pass'] and len(res['checks'])==24 and tests['success'] and tests['tests_run']==12 and ec['rank']==24 and ec['determinant']=='16'
result={'checked_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preservation':'PASS','original_files_checked':len(base),'original_files_unchanged':len(base)-len(changed),'original_modified_or_missing':changed,'old_audit_files_unchanged':sum(p.startswith('artifacts/audits/') for p in base),'previous_gate_files_unchanged':sum(p.startswith('artifacts/research_gates/') for p in base),'external_local_Particles_PDF_unchanged':not external,'new_files_outside_discovery_sandbox':extra,'HEAD':state['HEAD'].strip(),'branch':state['branch'].strip(),'status':state['status'].strip(),'all_recorded_refs_unchanged':True,'tracked_diff_exit':diff,'staged_diff_exit':staged,'reports':expected,'final_checks':{'action_algebra_assertions':24,'independent_tests':12,'EC_connection_rank':24,'EC_connection_det':16},'scope':'two candidates, one controlled physical-pairing revision; no automatic continuation; constitutive-predictivity freeze unchanged'}
(HERE/'preservation_final.json').write_text(json.dumps(result,indent=2))
# No cyclic hashes: omit this generated hash manifest and mutable timestamped verification result.
hashes={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['deliverable_sha256.json','preservation_final.json']}
(HERE/'deliverable_sha256.json').write_text(json.dumps(hashes,indent=2))
assert all(sha(OUT/p)==h for p,h in hashes.items())
assert all(sha(ROOT/p)==h for p,h in base.items())
print(json.dumps(result,indent=2))
