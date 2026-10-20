from pathlib import Path
import sys,os,subprocess,json,time,hashlib
out=Path(__file__).resolve().parents[1]; work=out/'working_copy'; ev=out/'evidence'
env=os.environ.copy(); overrides={'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(work),'MPLBACKEND':'Agg','MPLCONFIGDIR':str(out/'runtime/matplotlib'),'TMPDIR':str(out/'runtime/tmp')}; env.update(overrides)
cmd=[sys.executable,'-m','pytest',str(out/'tests'),'-q','-rs','--basetemp='+str(out/'runtime/pytest_independent')]; start=time.time()
p=subprocess.run(cmd,cwd=work,env=env,capture_output=True,text=True);(ev/'independent.stdout.txt').write_text(p.stdout);(ev/'independent.stderr.txt').write_text(p.stderr)
(ev/'independent_command.json').write_text(json.dumps({'command':cmd,'cwd':str(work),'environment_overrides':overrides,'exit_code':p.returncode,'seconds':time.time()-start,'test_sha256':hashlib.sha256((out/'tests/test_independent_audit.py').read_bytes()).hexdigest()},indent=2));print(p.stdout[-1800:]);sys.exit(p.returncode)
