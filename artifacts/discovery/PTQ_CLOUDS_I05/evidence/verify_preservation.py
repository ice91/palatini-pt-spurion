"""Hash originals read-only; write final evidence only inside I05."""
from pathlib import Path
import hashlib,json,subprocess,datetime
here=Path(__file__).resolve().parent
root=here.parents[3]
scope=str(here.parent.relative_to(root))+'/'
initial=json.loads((here/'initial_state.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
missing=[];changed=[]
for name,old in initial['original_sha256'].items():
 p=root/name
 if not p.is_file():missing.append(name)
 elif sha(p)!=old:changed.append(name)
added=[]
for p in root.rglob('*'):
 rel=p.relative_to(root)
 if '.git' in rel.parts or not p.is_file() or p.is_symlink():continue
 name=str(rel)
 if name not in initial['original_sha256'] and not name.startswith(scope):added.append(name)
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True)
state={'head':git('rev-parse','HEAD').strip(),'branch':git('branch','--show-current').strip(),
       'status':git('status','--short')}
same=all(initial[k]==state[k]for k in state)
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'original_count':len(initial['original_sha256']),'missing':missing,'changed':changed,
 'added_outside_sandbox':added,'git':state,'git_matches_initial':same,
 'pass':not(missing or changed or added) and same,
 'scope_note':'Regular project-file byte hashes; Git internals and symlink targets excluded.'}
(here/'preservation_final.json').write_text(json.dumps(result,indent=2)+'\n')
hashes={str(p.relative_to(here.parent)):sha(p)for p in sorted(here.parent.rglob('*'))
        if p.is_file()and p.name!='deliverable_sha256.json'}
(here/'deliverable_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps(result,indent=2))
if not result['pass']:raise SystemExit('PRESERVATION FAILURE')
