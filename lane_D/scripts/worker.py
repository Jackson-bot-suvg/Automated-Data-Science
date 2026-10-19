"""One isolated run; no offline oracle is imported into the agent process."""
import argparse, asyncio, json, os, random, sys, time, traceback
from common import *
sys.path[:0]=[str(base()),str(PKG/'src/adapters')]
from observe import Observer, BudgetExceeded
from methods import install
async def execute(spec,root):
    import numpy as np
    random.seed(spec['seed_block']);np.random.seed(spec['seed_block'])
    observer=Observer(root,spec['run_id'])
    if spec.get('observe',True):observer.install()
    install(spec['method_id'],observer)
    if spec.get('offline_smoke'):
        import app.llm
        class Disabled:
            async def chat(self,*a,**k):raise RuntimeError('OFFLINE_SMOKE_NO_API')
        app.llm.get_service=lambda:Disabled()
    from app.intake import build_initial_state
    from app.graph.workflow import build_workflow
    from app.schemas.state_schema import GraphState
    state=build_initial_state(task=spec['task'],dataset=spec['dataset'],workspace_root=str(root),thread_id=spec['run_id'])
    final=state.model_dump(mode='python');observer.emit('start',spec=spec)
    status='completed';t=time.monotonic()
    try:
        graph=build_workflow(stop_after_technical_summary=True)
        async for update in graph.astream(state,config={'recursion_limit':80},stream_mode='updates'):
            for phase,changes in update.items():
                final.update(changes)
                serial=GraphState.model_validate(final).model_dump(mode='json')
                dump(root/'final_state.json',serial)
                observer.emit('node_end',phase=phase,state_hash=digest(serial),contract_digest=digest(serial['target_spec']),target_spec=serial['target_spec'],review_state=serial['review_state'],model_state=serial['model_state'])
    except BudgetExceeded:
        status='budget'
    except BaseException as e:
        status='exception'
        # Traceback frames only: avoid provider messages that can contain credentials.
        (root/'exception.txt').write_text(''.join(traceback.format_tb(e.__traceback__))+'\n'+type(e).__name__)
    dump(root/'execution.json',dict(status=status,wall_seconds=time.monotonic()-t,api_calls=observer.calls,run_id=spec['run_id'],offline_smoke=spec.get('offline_smoke',False)))
    observer.emit('end',status=status)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--spec',required=True);args=a.parse_args()
    spec=json.loads(Path(args.spec).read_text());root=Path(spec['run_dir']).resolve();root.mkdir(parents=True,exist_ok=True)
    os.chdir(root);asyncio.run(execute(spec,root))
