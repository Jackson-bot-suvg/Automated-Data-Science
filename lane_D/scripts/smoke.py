"""Offline workflow/hook equivalence only; NEVER contributes to natural-risk denominators."""
import json, os, subprocess, sys, uuid
from common import *
def main():
    verify=subprocess.run([sys.executable,'-m','pytest','-c',str(PKG/'pytest.ini'),str(PKG/'tests'),'-q'],cwd=PKG)
    if verify.returncode:raise SystemExit(verify.returncode)
    results=[]
    for task,dataset in [('predict survival','titanic_autokaggle'),('classify creature type','ghouls_goblins_and_ghosts_boo')]:
        signatures=[]
        for observed in [False,True]:
            root=PKG/'tmp/smoke'/f'{dataset}-{observed}-{uuid.uuid4().hex[:6]}'
            root.mkdir(parents=True)
            spec=dict(run_id=root.name,run_dir=str(root),task=task,dataset=str(base()/'data'/dataset),method_id='F0_ORIGINAL',seed_block=1101,observe=observed,offline_smoke=True)
            dump(root/'spec.json',spec)
            env={k:v for k,v in os.environ.items() if k in ['PATH','HOME','TMPDIR','LANG']}
            env.update(LANE_D_BASE=str(base()),LANGSMITH_TRACING='false',LANGCHAIN_TRACING_V2='false',MPLBACKEND='Agg',PYTHONHASHSEED='1101',OMP_NUM_THREADS='2')
            with (root/'stdout.log').open('w') as log:
                r=subprocess.run([sys.executable,str(PKG/'scripts/worker.py'),'--spec',str(root/'spec.json')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            if r.returncode:raise RuntimeError(f'Worker failed: {root}')
            ex=json.loads((root/'execution.json').read_text());state=json.loads((root/'final_state.json').read_text())
            pred=state['model_state']['prediction_path'];predpath=root/pred
            signature=dict(execution_status=ex['status'],target_columns=state['target_spec']['target_columns'],primary_metric=state['target_spec']['evaluation']['primary_metric'],review_status=state['review_state']['latest_review_status'],artifact_sha=sha(predpath) if pred and predpath.is_file() else 'MISSING',best_model=state['model_state']['best_model_name'])
            signatures.append(signature)
            results.append(dict(dataset=dataset,observed=observed,path=str(root.relative_to(PKG)),signature=signature))
            print(dataset,observed,signature,flush=True)
        # Same termination alone is not enough; require an actual submission and exact prediction bytes.
        if signatures[0]!=signatures[1] or signatures[0]['artifact_sha']=='MISSING' or signatures[0]['execution_status']!='completed':
            dump(PKG/'results/smoke.json',dict(status='FAIL',runs=results));raise SystemExit('Hook equivalence failed')
    dump(PKG/'results/smoke.json',dict(status='PASS',type='offline deterministic fallback; no model API',runs=results))
    verification=subprocess.run([sys.executable,'-m','pytest','-c',str(PKG/'pytest.ini'),str(PKG/'tests'),'-q'],cwd=PKG,capture_output=True,text=True)
    (PKG/'results/unit_tests.txt').write_text(verification.stdout+verification.stderr)
    if verification.returncode:raise SystemExit(verification.returncode)
    print('PASS: two original datasets; hook off/on predictions identical; no API calls; excluded from study.')
if __name__=='__main__':main()
