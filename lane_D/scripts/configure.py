"""Write prospective protocol metadata from original source documentation."""
import sys,json,platform,os,csv
from common import *
sys.path.insert(0,str(base()))
from app.validation_matrix import ALL_VALIDATION_DATASETS
reg={}
for d in ALL_VALIDATION_DATASETS:
    tid='DEV-'+d.key.upper().replace('_','-');targets=d.target.split(',')
    with (base()/d.dataset/'sample_submission.csv').open() as f:cols=next(csv.reader(f))
    metric='log_rmse' if d.key=='house_prices' else 'roc_auc' if d.key=='bank_churn' else 'roc_auc_macro_ovr' if d.key=='plate_defect' else 'accuracy'
    reg[tid]=dict(dataset=d.dataset,target_columns=targets,id_columns=[c for c in cols if c not in targets],primary_metric=metric,prediction_kind='probability' if d.key in ['bank_churn','plate_defect'] else 'continuous' if d.key=='house_prices' else 'label',evidence=d.dataset+'/overview.txt',evidence_sha=sha(base()/d.dataset/'overview.txt'),private_test_score='unavailable',scope='original task intent; not private label oracle')
dump(PKG/'protocol/oracle_registry.json',reg)
dump(PKG/'env/runtime_info.json',dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),cpu_count=os.cpu_count(),model_requested='deepseek/deepseek-flash',model_response_id='NOT_OBSERVED',api_calls=0))
