from pathlib import Path
import subprocess,json,hashlib,re
out=Path(__file__).resolve().parents[1]; ev=out/'evidence'; root=out.parents[2]; logs=[]
def run(args):
 p=subprocess.run(args,cwd=root,capture_output=True,text=True);logs.append({'command':args,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr});return p.stdout
run(['git','status','--porcelain=v1','--untracked-files=normal']);run(['git','remote','-v']);run(['git','show-ref']);run(['git','log','--all','--format=%H %aI %cI %s']);run(['git','reflog','--all','--format=%H %gD %gs']);run(['git','stash','list']);run(['git','rev-parse','--is-shallow-repository']);run(['git','fsck','--full','--no-reflogs','--unreachable'])
commits=run(['git','rev-list','--all','--reflog']).splitlines(); blobpaths={}; commit_trees=[]
for commit in commits:
 data=subprocess.check_output(['git','ls-tree','-r','-z',commit],cwd=root)
 entries=[]
 for e in data.split(b'\0'):
  if not e:continue
  spec,path=e.decode().split('\t',1);mode,kind,oid=spec.split();entries.append({'path':path,'kind':kind,'oid':oid})
  if kind=='blob':blobpaths.setdefault(oid,set()).add(path)
 commit_trees.append({'commit':commit,'entries':entries})
(ev/'git_commit_trees.json').write_text(json.dumps(commit_trees,indent=2));(ev/'git_commands.json').write_text(json.dumps(logs,indent=2))
blobdir=ev/'git_blobs';blobdir.mkdir(); index=[];hits=[]
pattern=re.compile(r'(?i)hessian|poisson|stueck|lagrange|second.variation|ADM|lambdify|\.diff\(|diff\(|mixing|m_T|mT|Lambda_mu|alpha1|alpha_1|determinant')
for oid,paths in blobpaths.items():
 data=subprocess.check_output(['git','cat-file','blob',oid],cwd=root)
 try:text=data.decode('utf-8')
 except UnicodeDecodeError:continue
 (blobdir/(oid+'.txt')).write_text(text)
 entry={'git_blob':oid,'paths':sorted(paths),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()};index.append(entry)
 matching=[{'line':i,'text':line[:500]} for i,line in enumerate(text.splitlines(),1) if pattern.search(line)]
 if matching:hits.append({**entry,'hits':matching})
(ev/'git_text_blob_index.json').write_text(json.dumps(index,indent=2));(ev/'git_search_hits.json').write_text(json.dumps(hits,indent=2))
# File histories relevant to source-to-physical-result steps.
paths=['palatini_pt/palatini/field_eq.py','palatini_pt/gw/quadratic_action.py','palatini_pt/gw/degeneracy.py','palatini_pt/gw/nlo.py','palatini_pt/equivalence/coeff_extractor.py','palatini_pt/equivalence/dbi_chain.py','palatini_pt/equivalence/closed_metric_chain.py','palatini_pt/equivalence/cspp_chain.py','scripts/mixing_matrix_extract.py','configs/coeffs/mixing_matrix_FRW.json','notebooks/10_c2_symbolic_demo.py','notebooks/10_c2_symbolic_demo.ipynb']
for path in paths:
 data=run(['git','log','--all','--reverse','--format=%H %aI %s','-p','--',path]);(ev/('history_'+path.replace('/','__')+'.txt')).write_text(data)
(ev/'git_commands.json').write_text(json.dumps(logs,indent=2))
print('commits',len(commits),'unique blobs',len(blobpaths),'UTF8 blobs',len(index),'all historical paths',len(set(p for ps in blobpaths.values() for p in ps)))
print('latex paths',sorted(set(p for ps in blobpaths.values() for p in ps if p.endswith(('.tex','.tikz','.sty','.cls')))))
for path in paths:
 versions=[{'blob':x['git_blob'],'bytes':x['bytes']} for x in index if path in x['paths']];print(path,versions)
