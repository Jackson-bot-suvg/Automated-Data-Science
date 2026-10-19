"""Read-only instrumentation. No gold contracts or labels are imported here."""
import functools, hashlib, inspect, json, threading, time
from pathlib import Path
from datetime import datetime, timezone
class BudgetExceeded(BaseException): pass  # cannot be swallowed by upstream Exception fallbacks
class Observer:
    def __init__(self,root,run_id):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.run_id=run_id;self.seq=0;self.lock=threading.RLock();self.calls=0
    def emit(self,event_type,**payload):
        with self.lock:
            self.seq+=1
            event=dict(event_id=f'{self.run_id}:{self.seq}',run_id=self.run_id,monotonic_seq=self.seq,timestamp=datetime.now(timezone.utc).isoformat(),phase=payload.pop('phase','runtime'),actor='observer',event_type=event_type,state_hash='NA',candidate_hash='NA',contract_digest='NA',source_fields=[],verdict='NA',next_action='NA',actual_fit_target_hash='NA',artifact_mapping_digest='NA',error_subtype='NA')
            event.update(payload)
            with (self.root/'events.jsonl').open('a') as f:f.write(json.dumps(event,default=str,ensure_ascii=False)+'\n')
    def frame(self,name,obj):
        import pandas as pd
        if obj is None:return {'sha256':'NA','columns':[]}
        if isinstance(obj,pd.Series):obj=obj.to_frame()
        elif not isinstance(obj,pd.DataFrame):obj=pd.DataFrame(obj)
        with self.lock:
            path=self.root/f'{name}-{self.seq+1}-{time.monotonic_ns()}.csv'
            obj.to_csv(path,index=True)
        return dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),columns=list(map(str,obj.columns)),rows=len(obj))
    def install(self):
        import app.agents.model_building as mb
        import app.tools.model_tools as mt
        from sklearn.pipeline import Pipeline
        original_fit=Pipeline.fit
        @functools.wraps(original_fit)
        def fit(pipe,X,y=None,*args,**kwargs):
            record=self.frame('actual-y',y)
            self.emit('actual_fit_start',actual_fit_target_hash=record['sha256'],y=record,estimator=repr(pipe.steps[-1][1]),x_columns=list(map(str,getattr(X,'columns',[]))))
            t=time.monotonic();result=original_fit(pipe,X,y,*args,**kwargs)
            self.emit('actual_fit_end',seconds=time.monotonic()-t,actual_fit_target_hash=record['sha256'])
            return result
        Pipeline.fit=fit
        for name in ['fit_and_select_model','predict_for_submission','build_submission']:
            original=getattr(mb,name)
            def wrap(fn,n):
                @functools.wraps(fn)
                def call(*args,**kwargs):
                    bound=inspect.signature(fn).bind(*args,**kwargs);bound.apply_defaults();a=bound.arguments
                    if n=='fit_and_select_model':
                        self.emit('fit_contract',primary_metric=a['primary_metric'],task_family=a['task_family'],y=self.frame('consumed-y',a['y_train']),random_state=a['random_state'])
                    elif n=='predict_for_submission':
                        self.emit('predict_contract',prediction_kind=a.get('prediction_kind'),task_family=a.get('task_family'),output_columns=a.get('output_columns'),x=self.frame('predict-x',a.get('X_test')))
                    else:
                        self.emit('serialize_input',test=self.frame('serialize-test',a.get('test_df')),predictions=self.frame('serialize-predictions',a.get('predictions')),id_columns=a.get('id_columns'))
                    result=fn(*args,**kwargs)
                    if n!='fit_and_select_model':self.emit(n+'_result',frame=self.frame(n,result))
                    return result
                return call
            replacement=wrap(original,name);setattr(mb,name,replacement);setattr(mt,name,replacement)
        from app.llm.service import LLMService
        original_chat=LLMService.chat
        async def chat(service,model,messages,**kwargs):
            self.calls+=1
            if self.calls>60:raise BudgetExceeded('60 model-call budget exceeded')
            self.emit('llm_request',requested_model=model,messages=[m.model_dump() for m in messages],parameters=kwargs)
            try:result=await original_chat(service,model,messages,**kwargs)
            except Exception as e:
                # Provider text may contain secrets; persist only exception class.
                self.emit('llm_error',error_subtype=type(e).__name__);raise
            self.emit('llm_response',response=result.model_dump() if hasattr(result,'model_dump') else str(type(result)))
            return result
        LLMService.chat=chat
