"""Read-only verification of all originals and new-write scope; writes only I07 evidence."""
from pathlib import Path
import os,json,hashlib,subprocess
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
SANDBOX=OUT.parent
state=json.loads((OUT/'initial_state.json').read_text())
def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
changed=[]; missing=[]
for name,expected in state['sha256'].items():
    p=ROOT/name
    if not p.is_file(): missing.append(name)
    elif digest(p)!=expected: changed.append(name)
links_changed=[name for name,value in state['symlinks'].items() if not (ROOT/name).is_symlink() or os.readlink(ROOT/name)!=value]
new=[]
for base,dirs,names in os.walk(ROOT):
    dirs[:]=[d for d in dirs if d!='.git']
    for name in names:
        p=Path(base)/name; rel=str(p.relative_to(ROOT))
        if rel not in state['sha256'] and rel not in state['symlinks']: new.append(rel)
prefix=str(SANDBOX.relative_to(ROOT))+'/'
outside=[p for p in new if not p.startswith(prefix)]
final_git={'head':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),'status':git('status','--short')}
passed=not(changed or missing or links_changed or outside) and all(final_git[k]==state[k] for k in final_git)
result={'passed':passed,'protected_regular_file_count':len(state['sha256']),'changed_originals':changed,'missing_originals':missing,'changed_symlinks':links_changed,'new_paths_outside_I07':outside,'new_paths_inside_I07':sorted(new),'final_git':final_git,'git_matches_initial':all(final_git[k]==state[k] for k in final_git),'scope':'Regular files outside .git, plus symlink identities; all new writes must be under I07.'}
(OUT/'preservation_final.json').write_text(json.dumps(result,indent=2)+'\n')
manifest={str(p.relative_to(SANDBOX)):digest(p) for p in sorted(SANDBOX.rglob('*')) if p.is_file() and p.name!='deliverable_sha256.json'}
(OUT/'deliverable_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'passed':passed,'originals_verified':len(state['sha256']),'changed':len(changed),'missing':len(missing),'new_outside':outside,'git_matches':result['git_matches_initial'],'new_artifacts_hashed':len(manifest)},indent=2))
if not passed: raise SystemExit(1)
