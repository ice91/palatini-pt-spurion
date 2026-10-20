"""Verify I02 originals, scope, and executed evidence; write only I02 records."""
from pathlib import Path
from decimal import Decimal
import hashlib,json,subprocess,datetime
HERE=Path(__file__).resolve().parent;OUT=HERE.parent;ROOT=OUT.parents[2]
initial=json.loads((HERE/'initial_state.json').read_text())
assert str(ROOT)==initial['pwd']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base=initial['sha256'];prefix=str(OUT.relative_to(ROOT))+'/'
changed=[p for p,h in base.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
external=[p for p,h in initial['external_sha256'].items() if not Path(p).is_file() or sha(Path(p))!=h]
extra=[str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts
       and str(p.relative_to(ROOT)) not in base and not str(p.relative_to(ROOT)).startswith(prefix)]
now={k:subprocess.check_output(cmd,text=True,cwd=ROOT) for k,cmd in {
    'HEAD':['git','rev-parse','HEAD'],'branch':['git','branch','--show-current'],
    'status':['git','status','--short'],'refs':['git','show-ref']}.items()}
diff=subprocess.run(['git','diff','--exit-code'],cwd=ROOT,capture_output=True).returncode
staged=subprocess.run(['git','diff','--cached','--exit-code'],cwd=ROOT,capture_output=True).returncode
assert not changed and not external and not extra
assert all(now[k]==initial['git'][k] for k in now) and diff==staged==0
reports=['01_MODULAR_INFORMATION_CALCULATION.md','02_ENERGY_GEOMETRY_INTERFACE.md','03_ADVERSARIAL_AND_EXPLANATORY_VERDICT.md']
assert sorted(p.name for p in OUT.glob('*.md'))==reports
assert all(p.is_file() and p.name in reports or p.is_dir() and p.name=='evidence' for p in OUT.iterdir())
m=json.loads((HERE/'modular_results.json').read_text());g=json.loads((HERE/'geometry_results.json').read_text())
d=json.loads((HERE/'detector_results.json').read_text());i=json.loads((HERE/'independent_results.json').read_text())
assert m['all_pass'] and m['count']==len(m['checks'])==22
assert g['all_pass'] and g['count']==len(g['checks'])==9
assert d['all_pass'] and len(d['rows'])==2
assert i['success'] and i['tests']==10 and i['failures']==i['errors']==0
assert len(i['records']['detector_independent_convolution'])==2
for direct,independent in zip(d['rows'],i['records']['detector_independent_convolution']):
    assert direct['T']==independent['T']
    assert abs(Decimal(direct['state_excess'])-Decimal(independent['state_excess']))<Decimal('1e-35')
result={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preservation':'PASS',
        'original_files_checked':len(base),'modified_or_missing':changed,
        'external_original_files_checked':len(initial['external_sha256']),'external_changed':external,
        'new_files_outside_sandbox':extra,'HEAD':now['HEAD'].strip(),'branch':now['branch'].strip(),
        'status':now['status'].strip(),'refs_unchanged':True,'tracked_diff_exit':diff,'staged_diff_exit':staged,
        'prior_I01_files_preserved':sum(p.startswith('artifacts/discovery/PTQ_CLOUDS_I01/') for p in base),
        'prior_D02_files_preserved':sum(p.startswith('artifacts/discovery/PTQ_FOUNDATIONS_D02/') for p in base),
        'prior_audit_files_preserved':sum(p.startswith('artifacts/audits/') for p in base),
        'prior_gate_files_preserved':sum(p.startswith('artifacts/research_gates/') for p in base),
        'reports':reports,'executed_final':{'modular_checks':22,'modular_exact_symbolic_checks':21,
            'modular_packet_quadrature_checks':1,'geometry_checks':9,
            'independent_test_methods':10,'finite_detector_durations':2,'fixed_QFT_parent':1,
            'new_gravity_model':0,'empirical_fits':0},
        'decision':'information-stress identity and conditional geometric interface; no independent gravity law; IB-2 freeze CLOSED; STOP'}
(HERE/'preservation_final.json').write_text(json.dumps(result,indent=2))
hashes={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='deliverable_sha256.json'}
(HERE/'deliverable_sha256.json').write_text(json.dumps(hashes,indent=2))
assert all(sha(OUT/p)==h for p,h in hashes.items()) and all(sha(ROOT/p)==h for p,h in base.items())
print(json.dumps(result,indent=2))
