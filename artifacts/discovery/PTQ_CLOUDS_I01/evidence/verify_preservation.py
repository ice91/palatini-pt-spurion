"""Read-only project verification; all written evidence is confined to I01."""
from pathlib import Path
import hashlib,json,subprocess,datetime
HERE=Path(__file__).resolve().parent;OUT=HERE.parent;ROOT=OUT.parents[2]
initial=json.loads((HERE/'initial_state.json').read_text())
assert str(ROOT)==initial['pwd']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base=initial['sha256'];prefix=str(OUT.relative_to(ROOT))+'/'
changed=[p for p,h in base.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
external=[p for p,h in initial['external_sha256'].items() if not Path(p).is_file() or sha(Path(p))!=h]
extra=[str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()
       and '.git' not in p.relative_to(ROOT).parts and str(p.relative_to(ROOT)) not in base
       and not str(p.relative_to(ROOT)).startswith(prefix)]
now={k:subprocess.check_output(cmd,text=True,cwd=ROOT) for k,cmd in {
    'HEAD':['git','rev-parse','HEAD'],'branch':['git','branch','--show-current'],
    'status':['git','status','--short'],'refs':['git','show-ref']}.items()}
diff=subprocess.run(['git','diff','--exit-code'],cwd=ROOT,capture_output=True).returncode
staged=subprocess.run(['git','diff','--cached','--exit-code'],cwd=ROOT,capture_output=True).returncode
assert not changed and not external and not extra
assert all(now[k]==initial['git'][k] for k in now) and diff==staged==0
reports=['01_PHYSICAL_PROBLEM_AND_AXIOMS.md','02_EXACT_OBSERVER_REDUCTION.md',
         '03_INFORMATION_CAUSALITY_AND_TIME.md','04_EXPLANATORY_VERDICT.md']
assert sorted(p.name for p in OUT.glob('*.md'))==reports
assert all(p.is_file() and p.name in reports or p.is_dir() and p.name=='evidence' for p in OUT.iterdir())
symbolic=json.loads((HERE/'symbolic_results.json').read_text())
independent=json.loads((HERE/'independent_results.json').read_text())
assert symbolic['all_pass'] and symbolic['count']==len(symbolic['checks'])==26
assert independent['success'] and independent['tests']==9 and independent['failures']==independent['errors']==0
rows=independent['records']['fourier_quadratures'];assert len(rows)==10
result={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preservation':'PASS',
        'original_files_checked':len(base),'modified_or_missing':changed,
        'external_original_files_checked':len(initial['external_sha256']),'external_changed':external,
        'new_files_outside_sandbox':extra,'HEAD':now['HEAD'].strip(),'branch':now['branch'].strip(),
        'status':now['status'].strip(),'refs_unchanged':True,'tracked_diff_exit':diff,'staged_diff_exit':staged,
        'original_D01_files_preserved':sum(p.startswith('artifacts/discovery/PTQ_FOUNDATIONS_D01/') for p in base),
        'original_D02_files_preserved':sum(p.startswith('artifacts/discovery/PTQ_FOUNDATIONS_D02/') for p in base),
        'original_audit_files_preserved':sum(p.startswith('artifacts/audits/') for p in base),
        'original_gate_files_preserved':sum(p.startswith('artifacts/research_gates/') for p in base),
        'reports':reports,'executed_final':{'symbolic_checks':26,'independent_test_methods':9,
        'fourier_quadratures':10,'parent_theories':1,'observer_protocols':2,'model_revisions':0,
        'substantive_cycles':2},'decision':'vacuum observer KMS thermality reproduced; no automatic information destruction or fundamental time arrow; predictivity freeze CLOSED; STOP'}
(HERE/'preservation_final.json').write_text(json.dumps(result,indent=2))
hashes={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='deliverable_sha256.json'}
(HERE/'deliverable_sha256.json').write_text(json.dumps(hashes,indent=2))
assert all(sha(OUT/p)==h for p,h in hashes.items())
assert all(sha(ROOT/p)==h for p,h in base.items())
print(json.dumps(result,indent=2))
