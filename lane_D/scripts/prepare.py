"""Generate a prospective plan. Never infer results from the plan."""
import sys, random
from common import *
sys.path.insert(0,str(base()))
from app.validation_matrix import ALL_VALIDATION_DATASETS
METHODS=['F0_ORIGINAL','M2_INTAKE_ONLY_STRONG','M0_NO_SEMANTIC_ENFORCEMENT']
SEEDS=[1101,1102]
def main():
    registry=[]; prompts=[]; files=[]; cases=[]
    for d in ALL_VALIDATION_DATASETS:
        source=base()/d.dataset
        taskid='DEV-'+d.key.upper().replace('_','-')
        package=[]
        for f in sorted(source.iterdir()):
            if f.is_file():
                h=sha(f);package.append((f.name,h));files.append(dict(task_id=taskid,path=str(f.relative_to(base())),sha256=h,bytes=f.stat().st_size))
        # Prompts paraphrase the original intent, never introduce an incorrect target/metric.
        descriptions=[d.task, f'Use the supplied training data to {d.task}. Produce predictions for the supplied test rows.',f'Complete this data-science task: {d.task}. Follow the original dataset documentation and sample submission when preparing the final CSV.']
        registry.append(dict(task_id=taskid,split_tier='DEV',source_group_id=d.key,source_name=d.key,source_url='https://github.com/Jackson-bot-suvg/Automated-Data-Science/tree/'+BASE_SHA+'/'+d.dataset,license='upstream data license not independently cleared; data excluded from ZIP',original_version=BASE_SHA,source_sha256=digest(package),task_family=d.task_family,target_columns=d.target,primary_metric='offline official overview audit; see oracle_registry.json',data_split_policy='original files unchanged; official test labels unavailable',split_hash=digest(package),owner_signed_at='UNSIGNED',applicability_notes='Steel: original multi-output reproduction only; not evidence of a new true-multilabel task' if d.key=='plate_defect' else 'Original manuscript DEV task'))
        for i,p in enumerate(descriptions):
            pid=f'P{i}';case=f'{taskid}-{pid}'
            prompts.append(dict(task_id=taskid,prompt_id=pid,task=p,prompt_sha=digest(p),dataset=d.dataset,target_hint='',metric_hint=''))
            cases.append(dict(case_id=case,task_id=taskid,template_id='C00',condition='natural_no_injection',injection_stage='none',base_sha256=digest(package),changed_paths='none',mutation_sha256='NA',visible_evidence_hash=digest([package,p]),C_gold_version='DEV-original-overview-v1',truth_status='visible authoritative documentation; no private test scores',information_level='I-A',recoverable='NA',expected_safe_action='correct_delivery',target_fault_fields='none',expected_detection_stage='NA',applicability='DEV',pre_registered_exclusion_reason=''))
    dump(PKG/'cases/prompts.json',prompts)
    writecsv(PKG/'cases/manifest.csv',cases);writecsv(PKG/'protocol/source_files.csv',files)
    writecsv(PKG/'protocol/task_registry_public.csv',registry)
    writecsv(PKG/'protocol/split_registry.csv',[dict(task_id=r['task_id'],split_hash=r['split_hash'],private_test_labels='unavailable',policy=r['data_split_policy']) for r in registry])
    plan=[]
    for seed in SEEDS:
        block=[]
        for p in prompts:
            for method in METHODS:
                cell=f"D0DEV-{p['task_id']}-{p['prompt_id']}-{seed}-{method.split('_')[0]}"
                block.append(dict(plan_cell_id=cell,lane_id='D',stage='D0_DEV',task_id=p['task_id'],source_group_id=next(r['source_group_id'] for r in registry if r['task_id']==p['task_id']),condition='natural_no_injection',template_id='C00',case_id=p['task_id']+'-'+p['prompt_id'],method_id=method,host_id='RTS_LOCAL',model_id='deepseek/deepseek-flash',seed_block=seed,planned_status='not_started',applicability='planned',n_a_reason='',freeze_sha=sha(PKG/'FREEZE_SIGNOFF.yaml'),prompt_id=p['prompt_id'],order_index=0))
        random.Random(20261019+seed).shuffle(block);plan+=block
    for i,row in enumerate(plan):row['order_index']=i
    old={r['plan_cell_id']:r for r in readcsv(PKG/'logs/plan_ledger.csv')}
    for row in plan:
        if row['plan_cell_id'] in old:row['planned_status']=old[row['plan_cell_id']]['planned_status']
    writecsv(PKG/'logs/plan_ledger.csv',plan)
    future=[]
    for family in ['BIN','MC','ML','REG']:
        for n in range(1,4):
            for host in ['RTS_LOCAL','SECOND_HOST_UNSELECTED']:
                for seed in [2101,2102,2103]:future.append(dict(plan_cell_id=f'D0-CONF-{family}-{n}-{host}-{seed}',stage='D0_CONF',task_id=f'CONF-{family}-{n:02}',host_id=host,model_id='deepseek/deepseek-flash',condition='natural_no_injection',method_id='native_no_new_gates',seed_block=seed,status='not_started',reason='unsigned shared freeze; dataset/second host not registered'))
        for host in ['RTS_LOCAL','SECOND_HOST_UNSELECTED']:
            for model in ['deepseek/deepseek-flash','SECOND_FAMILY_UNSELECTED']:
                for condition in ['clean','unseen_intake','unseen_post_intake']:
                    for seed in [3101,3102,3103]:
                        for method in METHODS:future.append(dict(plan_cell_id=f'D1-{family}-{host}-{model}-{condition}-{seed}-{method}',stage='D1_TRANSFER',task_id=f'TR-{family}-01',host_id=host,model_id=model,condition=condition,method_id=method,seed_block=seed,status='not_started',reason='sealed tasks unavailable; transfer outside current no-injection scope'))
    writecsv(PKG/'logs/formal_plan_ledger.csv',future)
    print(f'{len(plan)} DEV cells; {len(future)} formal cells not started.')
if __name__=='__main__':main()
