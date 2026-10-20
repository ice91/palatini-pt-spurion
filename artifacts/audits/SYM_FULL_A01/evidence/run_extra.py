from pathlib import Path
import os,sys,subprocess,json,hashlib,shlex
out=Path(__file__).resolve().parents[1]; work=out/'working_copy'; ev=out/'evidence'; env=os.environ.copy(); env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(work),MPLBACKEND='Agg',MPLCONFIGDIR=str(out/'runtime/matplotlib'),TMPDIR=str(out/'runtime/tmp'))
commands=[('paper_test_module',[sys.executable,'-m','pytest','-q','-rs','--basetemp='+str(out/'runtime/pytest_paper')],{'PALPT_REQUIRE_REAL_APIS':'1'})]
results=[]
for name,cmd,extra in commands:
 e=env.copy(); e.update(extra); p=subprocess.run(cmd,cwd=work,env=e,capture_output=True,text=True); (ev/(name+'.stdout.txt')).write_text(p.stdout); (ev/(name+'.stderr.txt')).write_text(p.stderr); results.append({'label':name,'command':cmd,'extra':extra,'exit_code':p.returncode,'environment_overrides':{k:e[k] for k in ['PYTHONPATH','PYTHONDONTWRITEBYTECODE','MPLCONFIGDIR','TMPDIR']}}); print(name,p.returncode,p.stdout[-400:])
(ev/'extra_commands.json').write_text(json.dumps(results,indent=2))
