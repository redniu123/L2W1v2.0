#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse,csv,json,random,shlex,subprocess,sys
from datetime import datetime
from pathlib import Path
import numpy as np,yaml
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from modules.router.backfill import BackfillConfig,StrictBackfillController
from modules.router.circuit_breaker import CircuitBreaker,CircuitBreakerConfig
from modules.router.uncertainty_router import BudgetControllerConfig
from scripts.run_efficiency_frontier import build_agent_b_callable,ensure_agent_a_result_schema,infer_all_samples
from scripts.run_online_budget_control import run_online_pipeline

BR='GCR';BRV='router_v5.1';BRS='expA_run_20260413_run115631';SPV='dsp_v1.0';BV='backfill_v1_strict';CKV='breaker_v1';EV='eval_v1_expB';TOL=0.005

def jdump(p,o): p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def jlines(p,rows):
    with p.open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def wcsv(p,fs,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fs);w.writeheader();[w.writerow({k:r.get(k,'') for k in fs}) for r in rows]
def git(root):
    try:return subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(root),stderr=subprocess.DEVNULL,text=True).strip()
    except Exception:return 'unknown'
def rel(rows,shda):
    n=max(len(rows),1);c=lambda f:sum(1 for r in rows if f(r))
    h=c(lambda r:r.get('selected_for_upgrade') and r.get('edit_distance_final',0)>r.get('edit_distance_ocr',0) and abs(len(r.get('final_text',''))-len(r.get('ocr_text','')))>=3)
    fc=c(lambda r:r.get('selected_for_upgrade') and (r.get('final_text','').count('(')!=r.get('final_text','').count(')') or r.get('final_text','').count('[')!=r.get('final_text','').count(']')))
    oc=c(lambda r:r.get('selected_for_upgrade') and r.get('edit_distance_final',0)>r.get('edit_distance_ocr',0))
    fr=c(lambda r:r.get('backfill_status')=='rejected'); br=c(lambda r:shda and r.get('backfill_status')=='rejected'); tr=c(lambda r:shda and r.get('circuit_breaker_open'))
    return {'hallucination_rate':round(h/n,6),'format_collapse_rate':round(fc/n,6),'over_correction_rate':round(oc/n,6),'false_rejection_rate':round(fr/n,6),'backfill_reject_rate':round(br/n,6),'breaker_trigger_rate':round(tr/n,6),'hallucination_count':h,'format_collapse_count':fc,'over_correction_count':oc,'false_rejection_count':fr,'backfill_reject_count':br,'breaker_trigger_count':tr}
def load_rows(a):
    import argparse as _ap
    from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits
    from modules.router.domain_knowledge import DomainKnowledgeEngine
    cache=Path(a.cache_path)
    if a.use_cache and cache.exists() and not a.rebuild_cache:
        rows=ensure_agent_a_result_schema(json.loads(cache.read_text(encoding='utf-8')))
        return rows[:a.n_samples] if a.n_samples else rows
    ss=[json.loads(x) for x in Path(a.test_jsonl).read_text(encoding='utf-8').splitlines() if x.strip()]
    rec_args=_ap.Namespace(rec_model_dir=a.rec_model_dir,rec_char_dict_path=a.rec_char_dict_path,rec_image_shape='3, 48, 320',rec_batch_num=6,rec_algorithm='SVTR_LCNet',use_space_char=True,use_gpu=a.use_gpu,use_xpu=False,use_npu=False,use_mlu=False,use_metax_gpu=False,use_gcu=False,ir_optim=True,use_tensorrt=False,min_subgraph_size=15,precision='fp32',gpu_mem=500,gpu_id=0,enable_mkldnn=None,cpu_threads=10,warmup=False,benchmark=False,save_log_path='./log_output/',show_log=False,use_onnx=False,max_batch_size=10,return_word_box=False,drop_score=0.5,max_text_length=25,rec_image_inverse=True,use_det=False,det_model_dir='')
    rec=TextRecognizerWithLogits(rec_args);de=DomainKnowledgeEngine({'geology':a.geo_dict,'finance':a.finance_dict,'medicine':a.medicine_dict})
    rows=ensure_agent_a_result_schema(infer_all_samples(ss,rec,de,None,a.image_root));cache.parent.mkdir(parents=True,exist_ok=True);cache.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
    return rows[:a.n_samples] if a.n_samples else rows
def patch(rows,exp_id,name,b,run_id,pv,shda):
    for r in rows:r.update({'router_name':BR,'system_name':name,'exp_id':exp_id,'budget':b,'budget_mode':'online_control','run_id':run_id,'prompt_version':pv,'soft_prompt_version':SPV if shda else 'none','breaker_status':'triggered' if r.get('circuit_breaker_open') else 'not_triggered','breaker_reason':'high_rejection_rate' if r.get('circuit_breaker_open') else ''})
    return rows

def main():
    p=argparse.ArgumentParser(description='Run Main Experiment B')
    p.add_argument('--config',default='configs/router_config.yaml');p.add_argument('--test_jsonl',default='data/l2w1data/test.jsonl');p.add_argument('--image_root',default='data/l2w1data/images');p.add_argument('--rec_model_dir',default='./models/agent_a_ppocr/PP-OCRv5_server_rec_infer');p.add_argument('--rec_char_dict_path',default='ppocr/utils/ppocrv5_dict.txt');p.add_argument('--geo_dict',default='data/dicts/Geology.txt');p.add_argument('--finance_dict',default='data/dicts/Finance.txt');p.add_argument('--medicine_dict',default='data/dicts/Medicine.txt');p.add_argument('--output_dir',default='results/expriments/exB/02_runs');p.add_argument('--cache_path',default='results/expriments/exB/02_runs/agent_a_cache.json');p.add_argument('--budgets',nargs='+',type=float,default=[0.10,0.20,0.30]);p.add_argument('--seed',type=int,default=42);p.add_argument('--n_samples',type=int,default=None);p.add_argument('--use_gpu',action='store_true',default=False);p.add_argument('--use_cache',action='store_true',default=True);p.add_argument('--rebuild_cache',action='store_true',default=False);p.add_argument('--prompt_version',default=None);p.add_argument('--smoke_test',action='store_true',default=False)
    a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed)
    root=Path(__file__).resolve().parent.parent;cfg=yaml.safe_load(Path(a.config).read_text(encoding='utf-8'));pv=a.prompt_version or cfg.get('prompt_version') or cfg.get('mainline',{}).get('prompt_version','prompt_v1.1');cfg_run=json.loads(json.dumps(cfg))
    if a.smoke_test:a.n_samples=a.n_samples or 8;cfg_run.setdefault('agent_b',{})['skip']=True
    run_id=datetime.now().strftime('%Y%m%d_run%H%M%S');run_dir=Path(a.output_dir)/run_id;run_dir.mkdir(parents=True,exist_ok=True)
    rows=load_rows(a);agent_b=build_agent_b_callable(cfg_run);backfill=StrictBackfillController(BackfillConfig())
    snap={'run_id':run_id,'exp_group':'mainB','best_router_name':BR,'best_router_version':BRV,'best_router_source':BRS,'prompt_version':pv,'soft_prompt_version':SPV,'backfill_version':BV,'breaker_version':CKV,'eval_version':EV,'budget_points':a.budgets,'smoke_test':a.smoke_test,'args':vars(a)}
    (run_dir/'config_snapshot.yaml').write_text(yaml.safe_dump(snap,allow_unicode=True,sort_keys=False),encoding='utf-8');(run_dir/'command.sh').write_text('python '+' '.join(shlex.quote(x) for x in sys.argv)+'\n',encoding='utf-8')
    sums=[];metrics=[];all_rows=[];backfill_logs=[];breaker_logs=[];by={}
    for exp_id,name,strategy,shda,skip_backfill,disable_cb in [('M5','BestRouter-only','GCR',False,True,True),('M6','SH-DA++','GCR',True,False,False)]:
        for b in a.budgets:
            cb=CircuitBreaker(CircuitBreakerConfig(enabled=(False if disable_cb else True),min_samples=20,rejection_rate_threshold=0.60,cooldown_steps=50))
            class PromptStub:
                def generate_targeted_correction_prompt(self,**kw): return {'T_A':kw['T_A'],'min_conf_idx':kw.get('min_conf_idx'),'image_path':kw.get('image_path')}
            out=run_online_pipeline(strategy,b,rows,None,backfill,PromptStub(),agent_b,BudgetControllerConfig(target_budget=b),cb,run_id=run_id,prompt_version=pv,agent_b_label=cfg.get('mainline_agent_b','configured_agent_b'),skip_backfill=skip_backfill,domain_prompt=shda)
            ps=patch(out['per_sample'],exp_id,name,b,run_id,pv,shda);by[(exp_id,b)]=ps;all_rows+=ps;backfill_logs+=out.get('backfill_log',[]);breaker_logs+=[{'run_id':run_id,'exp_id':exp_id,'system_name':name,'sample_id':x.get('sample_id',''),'budget':b,'breaker_status':x.get('breaker_status','not_triggered'),'breaker_reason':x.get('breaker_reason',''),'degrade_action':'fallback_to_ocr' if x.get('breaker_status')=='triggered' else 'none'} for x in ps if shda]
            rr=rel(ps,shda);sums.append({'run_id':run_id,'exp_group':'mainB','exp_id':exp_id,'system_name':name,'best_router_name':BR,'budget':b,'budget_mode':'online_control','target_call_rate':b,'actual_call_rate':out['summary']['Actual_Call_Rate'],'call_rate_valid':abs(float(out['summary']['Actual_Call_Rate'])-b)<=0.005,'num_samples':out['summary']['N_valid'],'CER':out['summary']['Overall_CER'],'BoundaryDeletionRecallAtB':out['summary']['Boundary_Deletion_Recall@B'],'SubstitutionCER':out['summary']['Substitution_CER'],'CVR':out['summary']['CVR'],'AER':out['summary']['AER'],'p95_latency_ms':out['summary']['P95_Latency_MS'],'avg_token_usage':out['summary']['Avg_Token_Usage'],'data_version':'data_v20260413','split_version':'split_v1_test_frozen','agentA_version':'ocr_v5_server','agentB_model':cfg.get('agent_b',{}).get('model_name','unknown'),'model_version':cfg.get('agent_b',{}).get('model_name','unknown'),'best_router_version':BRV,'prompt_version':pv,'soft_prompt_version':SPV if shda else 'none','backfill_version':BV if shda else 'none','breaker_version':CKV if shda else 'none','eval_version':EV,'notes':'',**{k:v for k,v in rr.items() if k.endswith('_rate')}})
            metrics.append({'exp_id':exp_id,'system_name':name,'budget':b,'budget_mode':'online_control','target_call_rate':b,'actual_call_rate':out['summary']['Actual_Call_Rate'],'call_rate_valid':abs(float(out['summary']['Actual_Call_Rate'])-b)<=0.005,'num_samples':out['summary']['N_valid'],'metrics':{'cer':out['summary']['Overall_CER'],'boundary_deletion_recall_at_b':out['summary']['Boundary_Deletion_Recall@B'],'substitution_cer':out['summary']['Substitution_CER'],'cvr':out['summary']['CVR'],'aer':out['summary']['AER'],'p95_latency_ms':out['summary']['P95_Latency_MS'],'avg_token_usage':out['summary']['Avg_Token_Usage']},'reliability_metrics':{k:v for k,v in rr.items() if k.endswith('_rate')}})
            tag='BestRouterOnly' if exp_id=='M5' else 'SHDApp';jlines(run_dir/f'{exp_id}_{tag}_online_budget_{int(round(b*100))}_results.jsonl',ps)
    rel_rows=[{'run_id':run_id,'exp_id':s['exp_id'],'system_name':s['system_name'],'budget':s['budget'],'num_samples':len(by[(s['exp_id'],s['budget'])]),**rel(by[(s['exp_id'],s['budget'])],s['system_name']=='SH-DA++')} for s in sums]
    fail_rows=[{'run_id':run_id,'exp_id':x['exp_id'],'system_name':x['system_name'],'budget':x['budget'],'sample_id':x['sample_id'],'domain':x['domain'],'error_type':x['error_type'],'gt':x['gt'],'ocr_text':x['ocr_text'],'final_text':x['final_text'],'selected_for_upgrade':x['selected_for_upgrade'],'router_score':x['router_score'],'backfill_status':x['backfill_status'],'backfill_reason':x['backfill_reason'],'breaker_status':x.get('breaker_status',''),'breaker_reason':x.get('breaker_reason',''),'notes':''} for x in all_rows if (x.get('is_correct_ocr') is False and x.get('is_correct_final') is False)]
    budget_rows=[{'run_id':run_id,'exp_id':s['exp_id'],'system_name':s['system_name'],'budget':s['budget'],'target_call_rate':s['target_call_rate'],'actual_call_rate':s['actual_call_rate'],'abs_diff':round(abs(float(s['actual_call_rate'])-float(s['target_call_rate'])),4),'valid_threshold':TOL,'call_rate_valid':'valid' if abs(float(s['actual_call_rate'])-float(s['target_call_rate']))<=TOL else 'invalid','comment':''} for s in sums]
    wcsv(run_dir/'summary.csv',['run_id','exp_group','exp_id','system_name','best_router_name','budget','budget_mode','target_call_rate','actual_call_rate','call_rate_valid','num_samples','CER','BoundaryDeletionRecallAtB','SubstitutionCER','CVR','AER','p95_latency_ms','avg_token_usage','hallucination_rate','format_collapse_rate','over_correction_rate','false_rejection_rate','backfill_reject_rate','breaker_trigger_rate','data_version','split_version','agentA_version','agentB_model','model_version','best_router_version','prompt_version','soft_prompt_version','backfill_version','breaker_version','eval_version','notes'],sums)
    wcsv(run_dir/'reliability_breakdown.csv',list(rel_rows[0].keys()) if rel_rows else ['run_id'],rel_rows);wcsv(run_dir/'failure_cases.csv',['run_id','exp_id','system_name','budget','sample_id','domain','error_type','gt','ocr_text','final_text','selected_for_upgrade','router_score','backfill_status','backfill_reason','breaker_status','breaker_reason','notes'],fail_rows);wcsv(run_dir/'budget_check.csv',['run_id','exp_id','system_name','budget','target_call_rate','actual_call_rate','abs_diff','valid_threshold','call_rate_valid','comment'],budget_rows);wcsv(run_dir/'domain_breakdown.csv',['run_id','exp_id','system_name','best_router_name','budget','domain','num_samples','CER','BoundaryDeletionRecallAtB','SubstitutionCER','CVR','AER','hallucination_rate','format_collapse_rate','over_correction_rate','false_rejection_rate','avg_token_usage','p95_latency_ms'],[])
    jdump(run_dir/'metrics_summary.json',{'run_id':run_id,'exp_group':'mainB','best_router_name':BR,'best_router_version':BRV,'prompt_version':pv,'soft_prompt_version':SPV,'backfill_version':BV,'breaker_version':CKV,'eval_version':EV,'results':metrics});jlines(run_dir/'backfill_log.jsonl',backfill_logs);jlines(run_dir/'breaker_log.jsonl',breaker_logs);jdump(run_dir/'runtime_stats.json',{'run_id':run_id,'num_samples':len(rows),'budgets':a.budgets,'systems':['M5','M6'],'smoke_test':a.smoke_test,'total_rows':len(all_rows)})
    (run_dir/'run_manifest.md').write_text('# Run Manifest\n\n- run_id: %s\n- best_router_name: %s\n- budgets: %s\n- smoke_test: %s\n- status: done\n'%(run_id,BR,' / '.join(f'{b:.2f}' for b in a.budgets),a.smoke_test),encoding='utf-8');(run_dir/'env_snapshot.txt').write_text('python_version=%s\ngit_commit=%s\n'%(sys.version.split()[0],git(root)),encoding='utf-8');print(run_dir)

if __name__=='__main__': main()
