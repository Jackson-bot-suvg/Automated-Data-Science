"""Prospective, fail-closed orchestrator. Live mode requires an explicit environment file."""
import argparse, datetime, json, os, signal, subprocess, sys, time, uuid
from common import *
def verify_sources():
    rows=readcsv(PKG/'protocol/source_files.csv')
    if not rows:raise RuntimeError('Run prepare.py first')
    for r in rows:
        if sha(base()/r['path'])!=r['sha256']:raise RuntimeError('Source hash mismatch: '+r['path'])
    # The supplied main directory is an exported source tree, not a Git checkout.
    source_rows=readcsv(PKG/'protocol/base_code_files.csv')
    if not source_rows:raise RuntimeError('Missing base code manifest')
    expected={r['path'] for r in source_rows}
    actual={str(p.relative_to(base())) for p in (base()/'app').rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc')}
    if expected!=actual:raise RuntimeError('Base app file set differs from registered snapshot')
    for r in source_rows:
        if sha(base()/r['path'])!=r['sha256']:raise RuntimeError('Base code hash mismatch: '+r['path'])
def config_env():
    from dotenv import dotenv_values
    path=os.environ.get('LANE_D_ENV_FILE')
    if not path or not Path(path).is_file():raise RuntimeError('Live runs not started: set LANE_D_ENV_FILE to an existing credential file')
    values=dotenv_values(path)
    key=values.get('LLM_DEEPSEEK_API_KEY') or values.get('DEEPSEEK_API_KEY')
    if not key:raise RuntimeError('Credential file needs LLM_DEEPSEEK_API_KEY or DEEPSEEK_API_KEY')
    # Never forward arbitrary inherited tracing/proxy/API configuration.
    env={k:v for k,v in os.environ.items() if k in ['PATH','HOME','TMPDIR','LANG','LC_ALL']}
    env.update(LLM_PROVIDERS='deepseek',LLM_DEEPSEEK_API_KEY=key,LLM_DEEPSEEK_TYPE='openai_compat',LLM_DEEPSEEK_BASE_URL=values.get('LLM_DEEPSEEK_BASE_URL') or 'https://api.deepseek.com',LANGSMITH_TRACING='false',LANGCHAIN_TRACING_V2='false',MPLBACKEND='Agg',OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1',LANE_D_BASE=str(base()))
    for agent in ['READER','BG_REVIEWER','PLANNER','REVIEWER','SUMMARIZER','DATA_CLEANING','FEATURE_ENGINEERING','MODEL_BUILDING']:
        env[f'AGENT_{agent}_MODEL']='deepseek/deepseek-flash'
        env[f'AGENT_{agent}_TEMPERATURE']='0'
    return env

def launch(row,prompt,env):
    runid=row['plan_cell_id']+'-'+uuid.uuid4().hex[:10]
    root=PKG/'logs/runs'/runid;root.mkdir(parents=True,exist_ok=False)
    spec=dict(row,run_id=runid,seed_block=int(row['seed_block']),run_dir=str(root),dataset=str(base()/prompt['dataset']),task=prompt['task'])
    dump(root/'spec.json',spec)
    env=dict(env,PYTHONHASHSEED=str(row['seed_block']))
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (root/'stdout.log').open('w') as f:
        proc=subprocess.Popen([sys.executable,str(PKG/'scripts/worker.py'),'--spec',str(root/'spec.json')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:proc.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            dump(root/'execution.json',dict(status='budget',wall_seconds=1800,api_calls='NA',run_id=runid))
    if not (root/'execution.json').exists():dump(root/'execution.json',dict(status='infrastructure',wall_seconds='NA',api_calls='NA',run_id=runid,exit_code=proc.returncode))
    dump(root/'index.json',dict(run_id=runid,plan_cell_id=row['plan_cell_id'],started_at=started,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec_sha=sha(root/'spec.json'),path=str(root.relative_to(PKG))))
    return runid

def main():
    p=argparse.ArgumentParser();mode=p.add_mutually_exclusive_group(required=True)
    for flag in ['dry-run','pilot','resume','confirm']:mode.add_argument('--'+flag,action='store_true')
    p.add_argument('--limit',type=int);a=p.parse_args();verify_sources()
    rows=readcsv(PKG/'logs/plan_ledger.csv')
    if a.confirm:raise SystemExit('CONF/TRANSFER disabled: unsigned registry and sealed transfer inputs unavailable. Current runner is DEV only.')
    if a.dry_run:
        print(json.dumps(dict(dev_plan=len(rows),started=sum(r['planned_status']!='not_started' for r in rows),formal_plan=504,formal_started=0,models=['deepseek/deepseek-flash (unverified until API response)'],max_calls_per_run=60,max_wall_seconds_per_run=1800,call_upper_bound=len(rows)*60,sequential_wall_upper_hours=len(rows)*.5,estimated_cost='NA: no live pilot/provider pricing evidence',process_seeds=[1101,1102],training_seeds='original hardcoded seeds retained',prompt_variants=3),indent=2));return
    env=config_env();prompts=json.loads((PKG/'cases/prompts.json').read_text());count=0
    lock=PKG/'logs/.runner.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise SystemExit('Runner lock exists; verify no running process before manually removing it')
    try:
        os.write(fd,str(os.getpid()).encode());os.close(fd)
        for row in rows:
            if row['planned_status']!='not_started':continue
            # Mark before process start. Interrupted runs are never silently rerun by --resume.
            row['planned_status']='started';writecsv(PKG/'logs/plan_ledger.csv',rows)
            prompt=next(x for x in prompts if x['task_id']==row['task_id'] and x['prompt_id']==row['prompt_id'])
            runid=launch(row,prompt,env);row['planned_status']='finished';writecsv(PKG/'logs/plan_ledger.csv',rows)
            print(runid,flush=True);count+=1
            if a.limit and count>=a.limit:break
    finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
