import sys,json
from pathlib import Path
import pytest
P=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(P/'scripts'),str(P/'src/oracle'),str(P/'src/adapters')]
from common import base,readcsv
sys.path.insert(0,str(base()))
from judge import terminal,metric_matches
from methods import suppress_intake_semantics
from app.schemas.reviewer_schema import ReviewCheckResult
@pytest.mark.parametrize('kw,expected',[
    ({'artifact_valid':True,'semantic_correct':True},'S1'),
    ({'artifact_valid':True,'semantic_correct':True,'repaired':True},'S2'),
    ({'aborted':True,'justified':True},'S3'),({'aborted':True},'S4'),
    ({'artifact_valid':True,'semantic_correct':False,'exception':True},'S5'),
    ({'artifact_present':True},'S6'),({'exception':True},'S7'),
    ({'budget':True},'S8'),({'infra':True},'S8'),
    ({'artifact_valid':True,'semantic_correct':None},'S8'),
])
def test_terminal_priority(kw,expected):assert terminal(**kw)==expected

def test_no_semantic_ablation_does_not_destroy_structure():
    original=ReviewCheckResult(phase='reader',blocking_failures=['target unresolved: wrong','training dataset is missing at train.csv','submission mismatch: wrong order'])
    result=suppress_intake_semantics(original)
    assert result.blocking_failures==original.blocking_failures[1:]
    assert len(original.blocking_failures)==3

def test_metric_direction_and_log_scale():
    assert metric_matches('accuracy','accuracy')
    assert not metric_matches('roc_auc','accuracy')
    assert not metric_matches('neg_root_mean_squared_error','log_rmse')

def test_plan_has_no_injection_and_complete_pairs():
    rows=readcsv(P/'logs/plan_ledger.csv')
    assert len(rows)==144 and len({r['plan_cell_id'] for r in rows})==144
    assert {r['condition'] for r in rows}=={'natural_no_injection'}
    assert len({r['task_id'] for r in rows})==8
    for key in {(r['case_id'],r['seed_block']) for r in rows}:
        assert len([r for r in rows if (r['case_id'],r['seed_block'])==key])==3

def test_official_input_hashes():
    from run import verify_sources
    verify_sources()

def test_missing_key_fails_before_start(monkeypatch):
    from run import config_env
    monkeypatch.delenv('LANE_D_ENV_FILE',raising=False)
    with pytest.raises(RuntimeError,match='not started'):config_env()

def test_output_pairing_observer(tmp_path):
    import pandas as pd
    from observe import Observer
    obs=Observer(tmp_path,'test')
    frame=pd.DataFrame({'prediction':[.1,.8]},index=[42,7]);before=frame.copy(deep=True)
    record=obs.frame('test',frame);obs.emit('sample',frame=record)
    pd.testing.assert_frame_equal(frame,before)
    restored=pd.read_csv(tmp_path/record['path'],index_col=0)
    pd.testing.assert_frame_equal(restored,frame)
    assert json.loads((tmp_path/'events.jsonl').read_text())['monotonic_seq']==1

def test_zero_sample_intervals_are_undefined():
    from analyze import wilson,bootstrap
    assert wilson(0,0)==('NA','NA')
    assert bootstrap([])==('NA','NA')
    assert 0 < wilson(0,24)[1] < 1

def test_formal_grid_not_conflated_with_dev():
    formal=readcsv(P/'logs/formal_plan_ledger.csv')
    assert sum(r['stage']=='D0_CONF' for r in formal)==72
    assert sum(r['stage']=='D1_TRANSFER' for r in formal)==432
    assert {r['status'] for r in formal}=={'not_started'}

def test_observed_smoke_has_actual_action_chain():
    smoke_path=P/'results/smoke.json'
    if not smoke_path.exists():pytest.skip('Workflow smoke has not run yet')
    smoke=json.loads(smoke_path.read_text())
    for r in smoke['runs']:
        if not r['observed']:continue
        root=P/r['path']
        if not root.exists():pytest.skip('Archived smoke; rerun smoke to rebuild observation frames')
        es=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()]
        assert [e['monotonic_seq'] for e in es]==list(range(1,len(es)+1))
        types={e['event_type'] for e in es}
        assert {'fit_contract','actual_fit_start','predict_contract','serialize_input','build_submission_result'}.issubset(types)
        assert 'llm_response' not in types

def test_regression_numeric_alias():
    # Regression numeric/continuous are equivalent, unlike label/probability.
    from app.agents.reader import ReaderAgent
    assert ReaderAgent()._prediction_kind('regression',['price'],{}, {})=='numeric'

@pytest.mark.parametrize('dataset,taskid,expected',[('titanic_autokaggle','DEV-TITANIC-AUTOKAGGLE','S6'),('ghouls_goblins_and_ghosts_boo','DEV-GHOULS','S5')])
def test_offline_oracle_replays_actual_smoke(tmp_path,dataset,taskid,expected):
    import shutil
    from evaluate import judge_run
    smoke_path=P/'results/smoke.json'
    if not smoke_path.exists():pytest.skip('Run workflow smoke first')
    r=next(r for r in json.loads(smoke_path.read_text())['runs'] if r['dataset']==dataset and r['observed'])
    source=P/r['path']
    if not source.exists():pytest.skip('Regenerate nonredistributed source frames with --smoke')
    dest=tmp_path/'oracle_replay';shutil.copytree(source,dest)
    spec=json.loads((dest/'spec.json').read_text());spec.update(task_id=taskid,plan_cell_id='TEST_ONLY',source_group_id=dataset,case_id='TEST_ONLY',host_id='RTS_LOCAL',model_id='NO_API')
    (dest/'spec.json').write_text(json.dumps(spec))
    # Relative trace paths require root within package; test only substitutes relative_to for external tmp.
    import evaluate
    original=evaluate.PKG
    try:
        # Root need not be within PKG in production, so use actual package-local temp for this replay.
        local=P/'tmp'/'oracle_replay_tests'/dest.name
        if local.exists():shutil.rmtree(local)
        local.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(dest,local)
        result=judge_run(local)
        assert result['terminal_S']==expected
        assert result['model_id']=='unobserved'
    finally:
        evaluate.PKG=original

@pytest.mark.parametrize('mode,expected',[('intake','intake_misunderstanding'),('later','post_intake_state_change'),('action','state_action_mismatch')])
def test_first_drift_is_action_linked(mode,expected):
    from judge import first_drift
    # Synthetic unit fixtures only; not experimental injected cases or rate denominators.
    correct={'target_columns':['y'],'evaluation':{'primary_metric':'accuracy'}}
    wrong={'target_columns':['z'],'evaluation':{'primary_metric':'accuracy'}}
    events=[{'event_id':'reader:1','event_type':'node_end','phase':'reader','target_spec':wrong if mode=='intake' else correct}]
    if mode=='later':events.append({'event_id':'planner:2','event_type':'node_end','phase':'planner','target_spec':wrong})
    if mode=='action':events.append({'event_id':'fit:2','event_type':'fit_contract','y':{'columns':['z']},'primary_metric':'accuracy'})
    result=first_drift(events,{'target_columns':['y'],'primary_metric':'accuracy'})
    assert result[0]==expected and result[1]!='NA'
