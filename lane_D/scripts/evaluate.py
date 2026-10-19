"""Consumes completed traces offline; never modifies agent decisions."""
import sys,json
import numpy as np
import pandas as pd
from common import *
sys.path.insert(0,str(PKG/'src/oracle'))
from judge import terminal,metric_matches,first_drift
FIELDS='first_drift_type first_drift_event_id first_drift_field run_id plan_cell_id pair_id task_id source_group_id case_id case_family method_id implementation_sha split_hash oracle_version prompt_sha config_sha seed_block host_id model_id started_at finished_at terminal_S terminal_subreason semantic_correct artifact_valid actual_fit_target_hash actual_metric output_mapping_valid rejected_valid_case wall_seconds wrong_training_seconds api_calls tokens_in_if_observed tokens_out_if_observed measured_cost_if_observed artifact_sha256 event_trace_path exception_path missingness_reason'.split()
def events(root):
    p=root/'events.jsonl';return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def last(es,kind):return next((e for e in reversed(es) if e['event_type']==kind),None)
def frame(root,record):return pd.read_csv(root/record['path'],index_col=0)
def judge_run(root):
    s=json.loads((root/'spec.json').read_text());ex=json.loads((root/'execution.json').read_text());es=events(root)
    state=json.loads((root/'final_state.json').read_text()) if (root/'final_state.json').exists() else {}
    gold=json.loads((PKG/'protocol/oracle_registry.json').read_text())[s['task_id']]
    f=last(es,'fit_contract');pr=last(es,'predict_contract');ser=last(es,'serialize_input');built=last(es,'build_submission_result')
    path=state.get('model_state',{}).get('prediction_path','');path=Path(path) if path else root/'MISSING'
    if not path.is_absolute():path=root/path
    present=path.is_file();valid=False;semantic=None;mapping=None;reason=[]
    if present:
        out=pd.read_csv(path);rawtest=pd.read_csv(base()/gold['dataset']/'test.csv')
        valid=list(out.columns)==gold['id_columns']+gold['target_columns'] and len(out)==len(rawtest) and not out.isna().any().any()
        if valid:
            valid=all(out[c].astype(str).tolist()==rawtest[c].astype(str).tolist() for c in gold['id_columns'])
        if valid and gold['prediction_kind']=='label':
            rawtrain=pd.read_csv(base()/gold['dataset']/'train.csv',usecols=gold['target_columns'])
            valid=all(set(out[c].astype(str)).issubset(set(rawtrain[c].astype(str))) for c in gold['target_columns'])
        if valid and gold['prediction_kind']=='probability':valid=bool(((out[gold['target_columns']]>=0)&(out[gold['target_columns']]<=1)).all().all())
        if valid and all([f,pr,ser,built]):
            yf=frame(root,f['y']);pred=frame(root,ser['predictions']);ref=frame(root,ser['test']);px=frame(root,pr['x'])
            mapping=bool(list(px.index)==list(ref.index) and px.equals(ref[px.columns]) and np.array_equal(pred[gold['target_columns']].astype(str).values,out[gold['target_columns']].astype(str).values))
            fit_events=[e for e in es if e['event_type']=='actual_fit_start']
            actual_bound=bool(fit_events) and all(e['y']['columns']==gold['target_columns'] for e in fit_events)
            semantic=actual_bound and list(yf.columns)==gold['target_columns'] and metric_matches(f['primary_metric'],gold['primary_metric']) and mapping and (pr['prediction_kind']==gold['prediction_kind'] or (gold['prediction_kind']=='continuous' and pr['prediction_kind']=='numeric'))
            if not fit_events:semantic=None;reason.append('missing_actual_fit_observation')
            reason += ['metric_mismatch'] if not metric_matches(f['primary_metric'],gold['primary_metric']) else []
            reason += ['fit_target_mismatch'] if list(yf.columns)!=gold['target_columns'] else []
            reason += ['output_mapping_mismatch'] if not mapping else []
        elif valid:reason.append('missing_action_observation')
        else:reason.append('invalid_schema_rows_ids_or_value_domain')
    if valid and not all([f,pr,ser,built]):semantic=None
    aborted=state.get('review_state',{}).get('latest_review_status')=='fatal'
    outcome=terminal(artifact_present=present,artifact_valid=valid,semantic_correct=semantic,aborted=aborted,infra=ex['status']=='infrastructure' or (not ex.get('offline_smoke') and not any(e['event_type']=='llm_response' for e in es)),budget=ex['status']=='budget',exception=ex['status']=='exception')
    responses=[e['response'] for e in es if e['event_type']=='llm_response'];models=sorted({r.get('model','unknown') for r in responses});usage=[r['usage'] for r in responses if r.get('usage')]
    idx=json.loads((root/'index.json').read_text()) if (root/'index.json').exists() else {}
    row={k:'NA' for k in FIELDS};row.update({k:s[k] for k in ['run_id','plan_cell_id','task_id','source_group_id','case_id','method_id','seed_block','host_id']})
    row.update(pair_id=f"{s['case_id']}-{s['seed_block']}-{s['host_id']}-{s['model_id']}",case_family='natural_no_injection',implementation_sha=digest([(str(p.relative_to(PKG)),sha(p)) for p in sorted((PKG/'src').rglob('*.py'))]),oracle_version='DEV-original-overview-v1',prompt_sha=digest(s['task']),config_sha=sha(PKG/'protocol/frozen_config.yaml'),model_id=';'.join(models) or 'unobserved',started_at=idx.get('started_at','NA'),finished_at=idx.get('finished_at','NA'),terminal_S=outcome,terminal_subreason=';'.join(reason) or ex['status'],semantic_correct=semantic,artifact_valid=valid,actual_fit_target_hash=f['y']['sha256'] if f else 'NA',actual_metric=f['primary_metric'] if f else 'NA',output_mapping_valid=mapping,rejected_valid_case=outcome=='S4',wall_seconds=ex['wall_seconds'],api_calls=ex['api_calls'],tokens_in_if_observed=sum(u['input_tokens'] for u in usage) if usage else 'NA',tokens_out_if_observed=sum(u['output_tokens'] for u in usage) if usage else 'NA',artifact_sha256=sha(path) if present else 'NA',event_trace_path=str((root/'events.jsonl').relative_to(PKG)),exception_path=str((root/'exception.txt').relative_to(PKG)) if (root/'exception.txt').exists() else 'NA',missingness_reason='No private test score; cost not measured; LLM seed unsupported; original estimator seeds retained; semantic action audit is not model-quality validation')
    kind,eventid,field=first_drift(es,gold)
    row.update(first_drift_type=kind,first_drift_event_id=eventid,first_drift_field=field)
    return row

def main():
    if (PKG/'logs/.runner.lock').exists():raise SystemExit('Runner active or interrupted lock unresolved; do not score incomplete live traces')
    rows=[];index=[]
    for root in sorted((PKG/'logs/runs').glob('*')):
        if not (root/'spec.json').exists():continue
        if not (root/'execution.json').exists():dump(root/'execution.json',dict(status='infrastructure',wall_seconds='NA',api_calls='NA',run_id=root.name,reason='interrupted_before_execution_record'))
        if json.loads((root/'execution.json').read_text()).get('offline_smoke'):continue
        try:row=judge_run(root)
        except Exception as error:
            spec=json.loads((root/'spec.json').read_text());row={k:'NA' for k in FIELDS}
            row.update({k:spec[k] for k in ['run_id','plan_cell_id','task_id','source_group_id','case_id','method_id','seed_block','host_id']})
            row.update(terminal_S='S8',terminal_subreason='offline_observation_failure:'+type(error).__name__,pair_id=spec['case_id']+'-'+str(spec['seed_block'])+'-'+spec['host_id']+'-'+spec['model_id'],missingness_reason='Offline audit could not verify action chain; retain in denominator')
        rows.append(row);index.append(dict(run_id=root.name,path=str(root.relative_to(PKG)),sha256=sha(root/'execution.json'),status='evaluated'))
    writecsv(PKG/'results/run_level.csv',rows,FIELDS);writecsv(PKG/'logs/run_index.csv',index,['run_id','path','sha256','status'])
    writecsv(PKG/'results/drift_cases.csv',[{k:r[k] for k in ['run_id','terminal_S','first_drift_type','first_drift_event_id','first_drift_field']} for r in rows],['run_id','terminal_S','first_drift_type','first_drift_event_id','first_drift_field'])
    writecsv(PKG/'results/resource_ledger.csv',[{k:r[k] for k in ['run_id','method_id','wall_seconds','api_calls','tokens_in_if_observed','tokens_out_if_observed','measured_cost_if_observed','wrong_training_seconds']} for r in rows],['run_id','method_id','wall_seconds','api_calls','tokens_in_if_observed','tokens_out_if_observed','measured_cost_if_observed','wrong_training_seconds'])
    writecsv(PKG/'outputs/artifact_index.csv',[{k:r[k] for k in ['run_id','artifact_sha256','artifact_valid','terminal_subreason']} for r in rows],['run_id','artifact_sha256','artifact_valid','terminal_subreason'])
    print(f'{len(rows)} real runs evaluated; smoke results excluded')
if __name__=='__main__':main()
