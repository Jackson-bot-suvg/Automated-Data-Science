"""Aggregate all started runs; paired effects average within independent tasks first."""
import math,random,json
from common import *
def wilson(k,n):
    if not n:return ('NA','NA')
    z=1.95996398454;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h

def bootstrap(values):
    if len(values)<2:return ('NA','NA')
    rng=random.Random(1729);means=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(10000))
    return means[249],means[9749]

def main():
    rows=readcsv(PKG/'results/run_level.csv');groups={}
    for r in rows:groups.setdefault((r['task_id'],r['method_id']),[]).append(r)
    task=[]
    for (tid,m),rs in sorted(groups.items()):
        counts={f'S{i}':sum(r['terminal_S']==f'S{i}' for r in rs) for i in range(1,9)}
        task.append(dict(task_id=tid,method_id=m,n_started=len(rs),**counts,silent_error=counts['S5']/len(rs),correct_delivery=(counts['S1']+counts['S2'])/len(rs),unnecessary_abort=counts['S4']/len(rs)))
    writecsv(PKG/'results/task_level.csv',task,['task_id','method_id','n_started']+[f'S{i}' for i in range(1,9)]+['silent_error','correct_delivery','unnecessary_abort'])
    pairs=[];excluded=[]
    for pairid in sorted({r['pair_id'] for r in rows}):
        rr=[r for r in rows if r['pair_id']==pairid];f=[r for r in rr if r['method_id']=='F0_ORIGINAL'];m=[r for r in rr if r['method_id']=='M2_INTAKE_ONLY_STRONG']
        if len(f)!=1 or len(m)!=1:
            excluded.append(dict(pair_id=pairid,reason='missing_or_duplicate_method_run'));continue
        f,m=f[0],m[0]
        if f['model_id']!=m['model_id'] or f['model_id'] in ['NA','unobserved']:
            excluded.append(dict(pair_id=pairid,reason='actual_model_mismatch_or_unobserved'));continue
        for metric,states in [('silent_error',['S5']),('correct_delivery',['S1','S2']),('unnecessary_abort',['S4'])]:
            pairs.append(dict(pair_id=pairid,task_id=f['task_id'],metric=metric,F0_run=f['run_id'],M2_run=m['run_id'],delta=int(f['terminal_S'] in states)-int(m['terminal_S'] in states)))
    writecsv(PKG/'results/excluded_pairs.csv',excluded,['pair_id','reason'])
    writecsv(PKG/'results/pair_level.csv',pairs,['pair_id','task_id','metric','F0_run','M2_run','delta'])
    effects=[];loo=[]
    for metric in ['silent_error','correct_delivery','unnecessary_abort']:
        means={}
        for tid in {p['task_id'] for p in pairs}:
            vals=[p['delta'] for p in pairs if p['task_id']==tid and p['metric']==metric]
            if vals:means[tid]=sum(vals)/len(vals)
        values=list(means.values());lo,hi=bootstrap(values)
        effects.append(dict(metric=metric,n_tasks=len(values),estimate=sum(values)/len(values) if values else 'NA',ci_low=lo,ci_high=hi,unit='task; paired within description and process-seed block',status='DEV exploratory' if values else 'NOT_RUN'))
        for tid in means:
            remaining=[v for k,v in means.items() if k!=tid]
            loo.append(dict(metric=metric,omitted_task=tid,estimate=sum(remaining)/len(remaining) if remaining else 'NA'))
    writecsv(PKG/'paper/tables/T-D0-effects.csv',effects)
    writecsv(PKG/'paper/tables/T-D3-leave-one-out.csv',loo,['metric','omitted_task','estimate'])
    summary=[]
    for method in ['F0_ORIGINAL','M2_INTAKE_ONLY_STRONG','M0_NO_SEMANTIC_ENFORCEMENT']:
        rs=[r for r in rows if r['method_id']==method];n=len(rs);k=sum(r['terminal_S']=='S5' for r in rs);lo,hi=wilson(k,n)
        summary.append(dict(method_id=method,n_started=n,n_tasks=len({r['task_id'] for r in rs}),**{f'S{i}':sum(r['terminal_S']==f'S{i}' for r in rs) for i in range(1,9)},silent_rate=k/n if n else 'NA',descriptive_wilson_low=lo,descriptive_wilson_high=hi,note='run-level interval descriptive only; repeated tasks are dependent'))
    writecsv(PKG/'paper/tables/T-D0-outcomes.csv',summary)
    claims=[dict(claim_id='D0-NATURAL',experiment_id='D0_DEV',primary_contrast='F0 vs M2',estimand='natural silent semantic error',units='8 DEV tasks',estimate='NA' if not rows else 'see T-D0',ci_low='NA',ci_high='NA',qc_status='PENDING_LIVE_VALIDATION',result_usable='NO' if not rows else 'REQUIRES_TRACE_REVIEW',support_level='NOT_TESTED',counterexample_ids='NA',allowed_paper_wording='An API-free experimental harness has been prepared. No live natural-error rate or gate benefit has been established.' if not rows else 'DEV exploratory only; review actual traces before manuscript claims.'),dict(claim_id='D1-TRANSFER',experiment_id='D1',primary_contrast='F0 vs M2 across hosts/model families',estimand='transfer',units='0 opened tasks',estimate='NA',ci_low='NA',ci_high='NA',qc_status='BLOCKED_G0',result_usable='NO',support_level='NOT_TESTED',counterexample_ids='NA',allowed_paper_wording='No cross-host or cross-family transfer claim is supported.')]
    writecsv(PKG/'results/claims.csv',claims)
    print(f'{len(rows)} actual runs; {len(pairs)} metric-pairs; no planned cells included in outcome denominators.')
if __name__=='__main__':main()
