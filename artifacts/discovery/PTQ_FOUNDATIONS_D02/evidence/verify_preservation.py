"""Verify originals and all D02 results; write only D02 preservation/integrity records."""
from pathlib import Path
import hashlib,json,subprocess,datetime
HERE=Path(__file__).resolve().parent;OUT=HERE.parent;ROOT=OUT.parents[2]
initial=json.loads((HERE/'initial_state.json').read_text());assert str(ROOT)==initial['pwd']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base=initial['sha256'];prefix=str(OUT.relative_to(ROOT))+'/'
changed=[p for p,h in base.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
extra=[str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts and str(p.relative_to(ROOT)) not in base and not str(p.relative_to(ROOT)).startswith(prefix)]
now={k:subprocess.check_output(cmd,text=True,cwd=str(ROOT)) for k,cmd in {'HEAD':['git','rev-parse','HEAD'],'branch':['git','branch','--show-current'],'status':['git','status','--short'],'refs':['git','show-ref']}.items()}
diff=subprocess.run(['git','diff','--exit-code'],cwd=str(ROOT),capture_output=True).returncode
staged=subprocess.run(['git','diff','--cached','--exit-code'],cwd=str(ROOT),capture_output=True).returncode
assert not changed and not extra and all(now[k]==initial['git'][k] for k in now) and diff==staged==0
reports=['01_PARENT_MODEL_AND_DERIVATION.md','02_FIXED_PARENT_PROJECTION_TEST.md','03_PHYSICS_NOVELTY_AND_NEXT_DECISION.md']
assert sorted(p.name for p in OUT.glob('*.md'))==reports
assert all(p.is_dir() and p.name=='evidence' or p.is_file() and p.name in reports for p in OUT.iterdir())
geo=json.loads((HERE/'geometry_results.json').read_text());pro=json.loads((HERE/'projection_results.json').read_text());test=json.loads((HERE/'independent_test_results.json').read_text())
assert geo['all_pass'] and len(geo['checks'])==9 and geo['H0_rank']==24 and geo['H0_det']=='256'
assert pro['all_pass'] and len(pro['checks'])==19
assert test['success'] and test['tests']==11 and test['failures']==test['errors']==0
result={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preservation':'PASS','original_files_checked':len(base),'original_files_unchanged':len(base)-len(changed),'modified_or_missing':changed,'new_files_outside_D02':extra,'prior_audit_files_unchanged':sum(p.startswith('artifacts/audits/') for p in base),'prior_gate_files_unchanged':sum(p.startswith('artifacts/research_gates/') for p in base),'prior_D01_files_unchanged':sum(p.startswith('artifacts/discovery/PTQ_FOUNDATIONS_D01/') for p in base),'HEAD':now['HEAD'].strip(),'branch':now['branch'].strip(),'status':now['status'].strip(),'refs_unchanged':True,'tracked_diff_exit':diff,'staged_diff_exit':staged,'reports':reports,'final_execution':{'geometry_checks':9,'projector_response_checks':19,'independent_tests':11,'connection_H0_rank':24,'connection_H0_det':256,'parent_models':1,'candidate_retained_maps':2,'model_revisions':0,'substantive_cycles':2},'decision':'same representation class under matched full parent protocol; no PT-specific sector selection; no IB-2 rank result; STOPPED'}
(HERE/'preservation_final.json').write_text(json.dumps(result,indent=2))
hashes={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['preservation_final.json','deliverable_sha256.json']}
(HERE/'deliverable_sha256.json').write_text(json.dumps(hashes,indent=2))
assert all(sha(OUT/p)==h for p,h in hashes.items()) and all(sha(ROOT/p)==h for p,h in base.items())
print(json.dumps(result,indent=2))
