from pathlib import Path
import os,sys,json,hashlib,ast,collections,subprocess
import numpy as np
out=Path(__file__).resolve().parents[1]; root=out.parents[2]; work=out/'working_copy'; ev=out/'evidence'
os.chdir(work); sys.path.insert(0,str(work)); sys.dont_write_bytecode=True
os.environ.update(MPLBACKEND='Agg',MPLCONFIGDIR=str(out/'runtime/matplotlib'),PALPT_REQUIRE_REAL_APIS='1')
records={}; edges=collections.Counter()
def trace(frame,event,arg):
 if event=='call':
  fn=Path(frame.f_code.co_filename)
  if str(fn).startswith(str(work)):
   caller=frame.f_back
   edges[(str(Path(caller.f_code.co_filename)),caller.f_code.co_name,caller.f_lineno,str(fn.relative_to(work)),frame.f_code.co_name,frame.f_code.co_firstlineno)]+=1
 return trace
from palatini_pt.palatini.c1_pure_trace import run_c1
from palatini_pt.equivalence import compute_equivalence_report
from palatini_pt.equivalence.coeff_extractor import coeff_vectors
from palatini_pt.gw.tensor_mode import cT_of_k,waveform_overlay
from palatini_pt.gw.degeneracy import principal_eigs
from scripts.mixing_matrix_extract import load_mu_from_json,locking_ratios
from scripts.fig_c3_degeneracy import _eigs_via_module
from scripts.fig_c2_coeff_compare import _residual_scan_via_module
jobs={'C1':lambda:run_c1(np.array([.3,.1,-.2,.4])),'C2_public':lambda:compute_equivalence_report(alpha=-3),'C2_JSON':lambda:coeff_vectors(config=None,ibp_tol=1e-12),'C2_figure':lambda:_residual_scan_via_module({},np.array([1e-14,1e-9])),'C3_locked':lambda:cT_of_k(np.array([.01,.1]),{'box_eps':3},lock=True),'C3_matrix':lambda:locking_ratios(load_mu_from_json('configs/coeffs/mixing_matrix_FRW.json')),'C3_waveform':lambda:waveform_overlay(np.linspace(0,1,3),{}),'C3_figure_eigs':lambda:_eigs_via_module({'spurion':{'seps_scale':.2}}),'C3_direct_eigs':lambda:principal_eigs(config={},locked=True,n=40)}
for name,fn in jobs.items():
 edges.clear(); sys.setprofile(trace)
 try:r=fn(); result=str(r)
 except Exception as e:result=repr(e)
 finally:sys.setprofile(None)
 records[name]={'result':result,'edges':[{'caller_file':x[0],'caller':x[1],'caller_line':x[2],'file':x[3],'function':x[4],'definition_line':x[5],'calls':n} for x,n in edges.items()]}
(ev/'runtime_call_graphs.json').write_text(json.dumps(records,indent=2))
# Integrity sidecars use naked digest, not md5sum -c records.
def md5s(folder):
 rows=[]
 for p in sorted(folder.rglob('*.md5')):
  target=p.with_suffix(''); expected=p.read_text().strip().split()[0]; actual=hashlib.md5(target.read_bytes()).hexdigest() if target.exists() else None
  rows.append({'sidecar':str(p.relative_to(folder)),'expected':expected,'actual':actual,'match':expected==actual})
 return rows
original=md5s(root/'figs'); generated=md5s(work/'figs')
before=json.loads((ev/'artifacts_before.json').read_text()); after=json.loads((ev/'artifacts_after.json').read_text())
changes=[{'path':k,'before':before.get(k),'after':after.get(k),'same':before.get(k)==after.get(k)} for k in sorted(before.keys()|after.keys())]
result={'original_sidecars':original,'post_partial_make_sidecars':generated,'changes':changes,'original_eigenvalues':{'count':int(np.loadtxt(root/'figs/data/deg_eigvals.csv').size),'above_1e_10':int(np.sum(np.loadtxt(root/'figs/data/deg_eigvals.csv')>1e-10))},'generated_eigenvalues':np.loadtxt(work/'figs/data/deg_eigvals.csv').tolist()}
# compare numeric tables whenever formats agree
numeric=[]
for p in (work/'figs/data').glob('*.csv'):
 old=root/'figs/data'/p.name
 if old.exists():
  try:
   skip=0 if p.name=='deg_eigvals.csv' else 1
   a=np.loadtxt(old,delimiter=',',skiprows=skip); b=np.loadtxt(p,delimiter=',',skiprows=skip)
   numeric.append({'file':p.name,'original_shape':a.shape,'new_shape':b.shape,'exact_equal':np.array_equal(a,b),'max_abs_diff':float(np.max(np.abs(a-b))) if a.shape==b.shape else None})
  except Exception as e:numeric.append({'file':p.name,'reason':str(e)})
result['numeric_comparisons']=numeric
(ev/'artifact_consistency.json').write_text(json.dumps(result,indent=2)); print('sidecars',len(original),'original_bad',sum(not x['match'] for x in original),'generated_bad',sum(not x['match'] for x in generated)); print('changed',sum(not x['same'] for x in changes)); print(json.dumps(numeric,indent=2)); print('eigs',result['original_eigenvalues'],result['generated_eigenvalues'])
# Complete definition index for all relevant implementation, scripts, tests and notebook py.
index=[]
for top in ['palatini_pt','scripts','tests','notebooks']:
 for f in sorted((work/top).rglob('*.py')):
  tree=ast.parse(f.read_text()); defs=[{'name':node.name,'line':node.lineno,'kind':type(node).__name__} for node in ast.walk(tree) if isinstance(node,(ast.FunctionDef,ast.ClassDef))]
  index.append({'file':str(f.relative_to(work)),'definitions':defs})
(ev/'source_definition_index.json').write_text(json.dumps(index,indent=2))
