"""Offline rules only. S5 requires positive evidence, never absence of an exception."""
def terminal(*,artifact_present=False,artifact_valid=False,semantic_correct=None,repaired=False,justified=False,aborted=False,infra=False,budget=False,exception=False):
    if artifact_valid and semantic_correct is False:return 'S5'
    if infra or budget:return 'S8'
    if exception:return 'S7'
    if aborted:return 'S3' if justified else 'S4'
    if artifact_present and not artifact_valid:return 'S6'
    if artifact_valid and semantic_correct is True:return 'S2' if repaired else 'S1'
    # Observation failure is an infrastructure failure, not a fabricated success/error.
    if artifact_valid and semantic_correct is None:return 'S8'
    return 'S7'

def metric_matches(actual,expected):
    aliases={'accuracy':{'accuracy','accuracy_score'},'roc_auc':{'roc_auc','auc'},'roc_auc_macro_ovr':{'roc_auc_macro_ovr','roc_auc','auc'},'log_rmse':{'neg_root_mean_squared_log_error','rmsle','log_rmse'}}
    return str(actual).lower() in aliases.get(expected,{expected})

def first_drift(events,gold):
    """Return a candidate location for offline blind review, not a self-reported label."""
    def bad(spec):
        if list(spec.get('target_columns',[]))!=gold['target_columns']:return 'target_columns'
        if not metric_matches(spec.get('evaluation',{}).get('primary_metric',''),gold['primary_metric']):return 'primary_metric'
        return ''
    intake_seen=False
    for event in events:
        kind=event['event_type']
        if kind=='node_end' and event.get('target_spec'):
            field=bad(event['target_spec'])
            if event.get('phase')=='reader' and not intake_seen:
                intake_seen=True
                if field:return 'intake_misunderstanding',event['event_id'],field
            elif intake_seen and field:return 'post_intake_state_change',event['event_id'],field
        if kind=='fit_contract':
            if event['y']['columns']!=gold['target_columns']:return 'state_action_mismatch',event['event_id'],'fit_target'
            if not metric_matches(event['primary_metric'],gold['primary_metric']):return 'state_action_mismatch',event['event_id'],'fit_metric'
    return 'not_observed','NA','NA'
