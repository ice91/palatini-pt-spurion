import pathlib,subprocess,os,json,time,sys,hashlib
out=pathlib.Path(__file__).resolve().parents[1]; work=out/'working_copy'; ev=out/'evidence'
env=os.environ.copy(); env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(work),MPLBACKEND='Agg',MPLCONFIGDIR=str(out/'runtime/matplotlib'),TMPDIR=str(out/'runtime/tmp'))
pathlib.Path(env['TMPDIR']).mkdir(parents=True,exist_ok=True)
def hashes():
 return {str(p.relative_to(work)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (work/'figs').rglob('*') if p.is_file()}
(ev/'artifacts_before.json').write_text(json.dumps(hashes(),indent=2))
results=[]
for label,cmd,extra in [('baseline',[sys.executable,'-m','pytest','-q','-rs','--basetemp='+str(out/'runtime/pytest_baseline')],{}),('paper_test',['make','paper-test'],{'PALPT_REQUIRE_REAL_APIS':'1'}),('mixing',['make','mixing','PYTHON='+sys.executable],{}),('paper',['make','paper','PYTHON='+sys.executable],{})]:
 runenv=env.copy(); runenv.update(extra)
 # use the installed pytest associated with the same interpreter for make's bare pytest
 runenv['PATH']=str(pathlib.Path(sys.executable).parent)+':'+runenv.get('PATH','')
 start=time.time(); p=subprocess.run(cmd,cwd=work,env=runenv,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 (ev/f'{label}.stdout.txt').write_text(p.stdout); (ev/f'{label}.stderr.txt').write_text(p.stderr)
 result={'label':label,'command':cmd,'cwd':str(work),'environment_overrides':{k:runenv[k] for k in ['PYTHONPATH','PYTHONDONTWRITEBYTECODE','MPLBACKEND','MPLCONFIGDIR','TMPDIR','PATH']},'extra':extra,'exit_code':p.returncode,'seconds':time.time()-start}
 results.append(result); (ev/'baseline_commands.json').write_text(json.dumps(results,indent=2)); print(label,p.returncode,p.stdout[-2200:],p.stderr[-700:],flush=True)
(ev/'artifacts_after.json').write_text(json.dumps(hashes(),indent=2))
