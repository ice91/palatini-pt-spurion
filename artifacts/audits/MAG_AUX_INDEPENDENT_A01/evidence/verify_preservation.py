"""Read-only preservation comparison; writes its result only into this audit."""
import json,hashlib,subprocess
from pathlib import Path
E=Path(__file__).resolve().parent
root=E.parents[3]
out=E.parent
initial=json.loads((E/'initial_sha256.json').read_text())
missing=[];changed=[]
for name,digest in initial.items():
 p=root/name
 if not p.is_file():missing.append(name)
 elif hashlib.sha256(p.read_bytes()).hexdigest()!=digest:changed.append(name)
new=[]
for p in root.rglob('*'):
 if '.git' in p.relative_to(root).parts or out in p.parents or not p.is_file():continue
 name=str(p.relative_to(root))
 if name not in initial:new.append(name)
head=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True)
branch=subprocess.run(['git','branch','--show-current'],cwd=root,capture_output=True,text=True)
state=json.loads((E/'source_state.json').read_text())
result={'baseline_files':len(initial),'missing':missing,'changed':changed,'unexpected_new_outside_audit':new,'head':head.stdout.strip(),'branch':branch.stdout.strip(),'head_unchanged':head.stdout.strip()==state['head'],'branch_unchanged':branch.stdout.strip()==state['branch']}
result['PASS']=not missing and not changed and not new and result['head_unchanged'] and result['branch_unchanged']
(E/'preservation_final.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
raise SystemExit(0 if result['PASS'] else 1)
