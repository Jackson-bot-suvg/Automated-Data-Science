"""Rebuild evidence indexes; no figure of fabricated zero outcome rates."""
import json,shutil
from common import *
def main():
    for relative in ["paper/cases","paper/figures"]:(PKG/relative).mkdir(parents=True,exist_ok=True)
    smoke=json.loads((PKG/'results/smoke.json').read_text()) if (PKG/'results/smoke.json').exists() else {'status':'NOT_RUN','runs':[]}
    records=[]
    for r in smoke['runs']:
        records.append(dict(dataset=r['dataset'],observer=r['observed'],execution_status=r['signature']['execution_status'],prediction_sha=r['signature']['artifact_sha'],scope='OFFLINE_FALLBACK_NOT_LIVE',path=r['path']))
    writecsv(PKG/'paper/tables/T-D-preparation-smoke.csv',records,['dataset','observer','execution_status','prediction_sha','scope','path'])
    gates=[]
    for gid,status,why in [('G0','FAIL','Shared FREEZE_SIGNOFF unsigned; CONF/TRANSFER registry incomplete'),('G1','PARTIAL','Original file hashes registered; no private test labels'),('G2','PARTIAL','Original graph unchanged; F0/M2 scope overlap disclosed; live equivalence pending'),('G3','PARTIAL','Offline action observers verified; real-model traces absent'),('G4','PENDING','Actual response model identity and live budgets unverified'),('G5','PASS_PREPARATION','144 DEV and 504 formal slots preserved; 0 live started'),('G6','PENDING','No live result evidence or blinded adjudication'),('D0-OFFLINE',smoke['status'],'Two original tasks, observed/unobserved identical artifacts; smoke excluded from study'),('D0-QC','NOT_RUN','DeepSeek credentials pending; no live natural-error evidence'),('D1-QC','NOT_RUN','Sealed tasks, second host and second family unavailable')]:
        gates.append(dict(experiment_id='D',gate_id=gid,status=status,checked_count=4 if gid=='D0-OFFLINE' and smoke['status']=='PASS' else 0,eligible_count='NA',evidence_path='results/smoke.json' if gid=='D0-OFFLINE' else 'protocol/analysis_plan.md',reason=why,checked_by='automated preparation audit',checked_at='2026-10-19'))
    writecsv(PKG/'results/quality_gates.csv',gates)
    writecsv(PKG/'paper/tables/T-D2-unsupported.csv',[dict(host='second host',model='second family',stage='D1',status='not_implemented',reason='not available and no sealed tasks'),dict(host='RTS_LOCAL',model='deepseek/deepseek-flash',stage='D0_DEV',status='not_started',reason='credentials to be provided later')])
    (PKG/'paper/cases/NO_LIVE_CASES.md').write_text('No live supporting or counterexample case exists yet. Offline smoke traces establish execution compatibility only. Do not convert them into DeepSeek natural-drift claims.\n')
    (PKG/'paper/figures/README.md').write_text('No empirical effect plot is generated before live observations. Outcome and effect tables contain NA for unobserved rates, not zero.\n')
    print('Preparation tables and quality gates rebuilt.')
if __name__=='__main__':main()
