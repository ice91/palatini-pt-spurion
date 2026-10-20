"""Verify all pre-I03 files and Git state; write only I03 evidence."""
from pathlib import Path
import hashlib,json,subprocess,datetime
HERE=Path(__file__).resolve().parent;OUT=HERE.parent;ROOT=OUT.parents[2]
initial=json.loads((HERE/'initial_state.json').read_text())
assert str(ROOT)==initial['pwd']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base=initial['sha256'];prefix=str(OUT.relative_to(ROOT))+'/'
changed=[p for p,h in base.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
extra=[str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts
       and str(p.relative_to(ROOT)) not in base and not str(p.relative_to(ROOT)).startswith(prefix)]
now={k:subprocess.check_output(v,text=True,cwd=ROOT) for k,v in {
 'HEAD':['git','rev-parse','HEAD'],'branch':['git','branch','--show-current'],
 'status':['git','status','--short'],'refs':['git','show-ref']}.items()}
diff=subprocess.run(['git','diff','--exit-code'],cwd=ROOT,capture_output=True).returncode
staged=subprocess.run(['git','diff','--cached','--exit-code'],cwd=ROOT,capture_output=True).returncode
assert not changed and not extra and all(now[k]==initial['git'][k] for k in now) and diff==staged==0
reports=['01_MICROSCOPIC_AREA_ENTROPY.md','02_INDUCED_GRAVITATIONAL_COUPLING.md','03_FIRST_PRINCIPLES_GAP_AND_DECISION.md']
assert sorted(p.name for p in OUT.glob('*.md'))==reports
assert all(p.is_file() and p.name in reports or p.is_dir() and p.name=='evidence' for p in OUT.iterdir())
a=json.loads((HERE/'area_results.json').read_text());c=json.loads((HERE/'curvature_results.json').read_text());i=json.loads((HERE/'independent_results.json').read_text())
assert a['all_pass'] and a['count']==len(a['checks'])==14
assert c['all_pass'] and c['count']==len(c['checks'])==18
assert i['success'] and i['tests']==11 and i['failures']==i['errors']==0
attempt=json.loads((HERE/'area_attempt1.json').read_text())
assert attempt['success'] is False and attempt['model_or_expected_formula_changed'] is False
result={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preservation':'PASS',
 'original_files_checked':len(base),'modified_or_missing':changed,'new_files_outside_sandbox':extra,
 'HEAD':now['HEAD'].strip(),'branch':now['branch'].strip(),'status':now['status'].strip(),
 'refs_unchanged':True,'tracked_diff_exit':diff,'staged_diff_exit':staged,
 'prior_I01_files_preserved':sum(p.startswith('artifacts/discovery/PTQ_CLOUDS_I01/') for p in base),
 'prior_I02_files_preserved':sum(p.startswith('artifacts/discovery/PTQ_CLOUDS_I02/') for p in base),
 'prior_audit_files_preserved':sum(p.startswith('artifacts/audits/') for p in base),
 'prior_gate_files_preserved':sum(p.startswith('artifacts/research_gates/') for p in base),
 'reports':reports,'executed_final':{'area_exact_checks':14,'curvature_exact_checks':18,
 'independent_test_methods':11,'physical_field_models':1,'main_calculations':2,
 'physical_model_revisions':0,'initial_symbolic_normalization_failure_preserved':True},
 'decision':'minimal common-regulator entropy/induced-curvature matching; no finite predicted Newton constant, independent metric dynamics, or universal Clausius law; freeze CLOSED; STOP'}
(HERE/'preservation_final.json').write_text(json.dumps(result,indent=2))
hashes={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='deliverable_sha256.json'}
(HERE/'deliverable_sha256.json').write_text(json.dumps(hashes,indent=2))
assert all(sha(OUT/p)==h for p,h in hashes.items()) and all(sha(ROOT/p)==h for p,h in base.items())
print(json.dumps(result,indent=2))
