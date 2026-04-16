#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse,csv,json,random,subprocess,sys
from datetime import datetime
from pathlib import Path
import numpy as np,yaml
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from modules.router.backfill import BackfillConfig,StrictBackfillController
from modules.router.circuit_breaker import CircuitBreaker,CircuitBreakerConfig
from modules.router.uncertainty_router import BudgetControllerConfig
from scripts.run_efficiency_frontier import build_agent_b_callable,ensure_agent_a_result_schema,infer_all_samples,run_pipeline
from scripts.run_online_budget_control import run_online_pipeline
BR='GCR';SPV='dsp_v1.0';BV='backfill_v1_strict';CKV='breaker_v1';TOL=0.005
J=lambda p,o:p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def JL(p,rows):
    with p.open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def WC(p,fs,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fs);w.writeheader();[w.writerow({k:r.get(k,'') for k in fs}) for r in rows]
def G(root):
    try:return subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(root),stderr=subprocess.DEVNULL,text=True).strip()
    except Exception:return 'unknown'
def P(shda=False):
    class X:
        def generate_targeted_correction_prompt(self,**kw):
            d={'geology':'地质勘探','finance':'金融财会','medicine':'医学'}.get(kw.get('domain','geology')) if shda else None
            return {'T_A':kw['T_A'],'min_conf_idx':kw.get('min_conf_idx'),'image_path':kw.get('image_path'),'domain':d}
    return X()
def R(rows,shda):
    n=max(len(rows),1);c=lambda f:sum(1 for r in rows if f(r))
    return {'hallucination_rate':round(c(lambda r:r.get('selected_for_upgrade') and r.get('edit_distance_final',0)>r.get('edit_distance_ocr',0) and abs(len(r.get('final_text',''))-len(r.get('ocr_text','')))>=3)/n,6),'format_collapse_rate':round(c(lambda r:r.get('selected_for_upgrade') and (r.get('final_text','').count('(')!=r.get('final_text','').count(')') or r.get('final_text','').count('[')!=r.get('final_text','').count(']')))/n,6),'over_correction_rate':round(c(lambda r:r.get('selected_for_upgrade') and r.get('edit_distance_final',0)>r.get('edit_distance_ocr',0))/n,6),'false_rejection_rate':round(c(lambda r:r.get('backfill_status')=='rejected')/n,6),'backfill_reject_rate':round(c(lambda r:shda and r.get('backfill_status')=='rejected')/n,6),'breaker_trigger_rate':round(c(lambda r:shda and r.get('circuit_breaker_open'))/n,6)}
def L(a):
    cp=Path(a.cache_path)
    if a.use_cache and cp.exists() and not a.rebuild_cache:
        rows=ensure_agent_a_result_schema(json.loads(cp.read_text(encoding='utf-8')));return rows[:a.n_samples] if a.n_samples else rows
    import argparse as _ap
    from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits
    from modules.router.domain_knowledge import DomainKnowledgeEngine
    ss=[json.loads(x) for x in Path(a.test_jsonl).read_text(encoding='utf-8').splitlines() if x.strip()]
    ra=_ap.Namespace(rec_model_dir=a.rec_model_dir,rec_char_dict_path=a.rec_char_dict_path,rec_image_shape='3, 48, 320',rec_batch_num=6,rec_algorithm='SVTR_LCNet',use_space_char=True,use_gpu=a.use_gpu,use_xpu=False,use_npu=False,use_mlu=False,use_metax_gpu=False,use_gcu=False,ir_optim=True,use_tensorrt=False,min_subgraph_size=15,precision='fp32',gpu_mem=500,gpu_id=0,enable_mkldnn=None,cpu_threads=10,warmup=False,benchmark=False,save_log_path='./log_output/',show_log=False,use_onnx=False,max_batch_size=10,return_word_box=False,drop_score=0.5,max_text_length=25,rec_image_inverse=True,use_det=False,det_model_dir='')
    rec=TextRecognizerWithLogits(ra);de=DomainKnowledgeEngine({'geology':a.geo_dict,'finance':a.finance_dict,'medicine':a.medicine_dict});rows=ensure_agent_a_result_schema(infer_all_samples(ss,rec,de,None,a.image_root));cp.parent.mkdir(parents=True,exist_ok=True);cp.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8');return rows[:a.n_samples] if a.n_samples else rows
def cache(rows,cfg,bf,rd,pv,shda,path,rebuild):
    p=Path(path) if path else (rd/('m6_full_call_cache.jsonl' if shda else 'm5_full_call_cache.jsonl'));p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and not rebuild:return p,[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
    ag=build_agent_b_callable(cfg);strategy='SH-DA++' if shda else 'Router-only';o=run_pipeline(strategy,1.0,rows,None,bf,P(shda),ag,run_id=rd.name,prompt_version=pv,agent_b_label=getattr(ag,'_model_label','configured_agent_b'))
    rs=[{'sample_id':x.get('sample_id',''),'image_path':x.get('image_path',''),'ocr_text':x.get('ocr_text',''),'vlm_raw_output':x.get('vlm_raw_output',''),'latency_ms':x.get('latency_ms'),'token_usage':x.get('token_usage'),'error_type':x.get('error_type','none'),'vlm_model':x.get('vlm_model',''),'prompt_version':x.get('prompt_version',pv),'cache_variant':'M6' if shda else 'M5'} for x in o['per_sample']];JL(p,rs);return p,rs
def A(rows,label):
    bp={};bi={};ml=label
    for r in rows:bp.setdefault((r.get('image_path',''),r.get('ocr_text','')),r);bi.setdefault(r.get('image_path',''),r);ml=r.get('vlm_model') or ml
    def call(prompt):
        h=bp.get((prompt.get('image_path',''),prompt.get('T_A',''))) or bi.get(prompt.get('image_path',''))
        return {'corrected_text':(h or {}).get('vlm_raw_output',prompt.get('T_A','')) or prompt.get('T_A',''),'latency_ms':0.0,'token_usage':(h or {}).get('token_usage'),'error_type':(h or {}).get('error_type','cache_miss' if h is None else 'none')}
    call._model_label=ml;call._supports_parallel=False;call._max_concurrency=1;return call
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Main Experiment B dual-cache');p.add_argument('--config',default='configs/router_config.yaml');p.add_argument('--test_jsonl',default='data/l2w1data/test.jsonl');p.add_argument('--image_root',default='data/l2w1data/images');p.add_argument('--rec_model_dir',default='./models/agent_a_ppocr/PP-OCRv5_server_rec_infer');p.add_argument('--rec_char_dict_path',default='ppocr/utils/ppocrv5_dict.txt');p.add_argument('--geo_dict',default='data/dicts/Geology.txt');p.add_argument('--finance_dict',default='data/dicts/Finance.txt');p.add_argument('--medicine_dict',default='data/dicts/Medicine.txt');p.add_argument('--output_dir',default='results/expriments/exB/02_runs');p.add_argument('--cache_path',default='results/expriments/exB/02_runs/agent_a_cache.json');p.add_argument('--m5_cache_path',default=None);p.add_argument('--m6_cache_path',default=None);p.add_argument('--reuse_cached_vlm',action='store_true');p.add_argument('--prepare_dual_cache_only',action='store_true');p.add_argument('--rebuild_m5_cache',action='store_true');p.add_argument('--rebuild_m6_cache',action='store_true');p.add_argument('--budgets',nargs='+',type=float,default=[0.10,0.20,0.30]);p.add_argument('--seed',type=int,default=42);p.add_argument('--n_samples',type=int);p.add_argument('--use_gpu',action='store_true');p.add_argument('--use_cache',action='store_true',default=True);p.add_argument('--rebuild_cache',action='store_true');p.add_argument('--prompt_version',default=None);p.add_argument('--smoke_test',action='store_true');a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed)
    root=Path(__file__).resolve().parent.parent;cfg=yaml.safe_load(Path(a.config).read_text(encoding='utf-8'));pv=a.prompt_version or cfg.get('prompt_version') or cfg.get('mainline',{}).get('prompt_version','prompt_v1.1');cfg2=json.loads(json.dumps(cfg));
    if a.smoke_test:a.n_samples=a.n_samples or 8;cfg2.setdefault('agent_b',{})['skip']=True
    rid=datetime.now().strftime('%Y%m%d_run%H%M%S');rd=Path(a.output_dir)/rid;rd.mkdir(parents=True,exist_ok=True);rows=L(a);bf=StrictBackfillController(BackfillConfig())
    if a.prepare_dual_cache_only or a.reuse_cached_vlm:m5p,m5r=cache(rows,cfg2,bf,rd,pv,False,a.m5_cache_path,a.rebuild_m5_cache);m6p,m6r=cache(rows,cfg2,bf,rd,pv,True,a.m6_cache_path,a.rebuild_m6_cache);m5a=A(m5r,'cached_m5_full_call');m6a=A(m6r,'cached_m6_full_call')
    else:m5p=m6p='';real=build_agent_b_callable(cfg2);m5a=real;m6a=real
    if a.prepare_dual_cache_only:J(rd/'config_snapshot.yaml',{'run_id':rid,'mode':'prepare_dual_cache_only','m5_cache_path':str(m5p),'m6_cache_path':str(m6p),'prompt_version':pv});print(f'M5 cache ready: {m5p}');print(f'M6 cache ready: {m6p}');print(rd);sys.exit(0)
    sums=[];mets=[];allr=[];bfl=[];brl=[];by={}
    for eid,name,shda,skipbf,discb,ag in [('M5','BestRouter-only',False,True,True,m5a),('M6','SH-DA++',True,False,False,m6a)]:
        for b in a.budgets:
            o=run_online_pipeline('GCR',b,rows,None,bf,P(shda),ag,BudgetControllerConfig(target_budget=b),CircuitBreaker(CircuitBreakerConfig(enabled=(False if discb else True),min_samples=20,rejection_rate_threshold=0.60,cooldown_steps=50)),run_id=rid,prompt_version=pv,agent_b_label=getattr(ag,'_model_label',cfg.get('mainline_agent_b','configured_agent_b')),skip_backfill=skipbf,domain_prompt=shda);ps=o['per_sample']
            [r.update({'router_name':BR,'system_name':name,'exp_id':eid,'budget':b,'budget_mode':'online_control','run_id':rid,'prompt_version':pv,'soft_prompt_version':SPV if shda else 'none','breaker_status':'triggered' if r.get('circuit_breaker_open') else 'not_triggered'}) for r in ps];by[(eid,b)]=ps;allr+=ps;bfl+=o.get('backfill_log',[]);rr=R(ps,shda);note='reuse_m5_cache' if (a.reuse_cached_vlm and eid=='M5') else ('reuse_m6_cache' if (a.reuse_cached_vlm and eid=='M6') else '')
            sums.append({'run_id':rid,'exp_id':eid,'system_name':name,'budget':b,'target_call_rate':b,'actual_call_rate':o['summary']['Actual_Call_Rate'],'call_rate_valid':abs(float(o['summary']['Actual_Call_Rate'])-b)<=TOL,'num_samples':o['summary']['N_valid'],'CER':o['summary']['Overall_CER'],'BoundaryDeletionRecallAtB':o['summary']['Boundary_Deletion_Recall@B'],'SubstitutionCER':o['summary']['Substitution_CER'],'CVR':o['summary']['CVR'],'AER':o['summary']['AER'],'p95_latency_ms':o['summary']['P95_Latency_MS'],'avg_token_usage':o['summary']['Avg_Token_Usage'],'agentB_model':getattr(ag,'_model_label','unknown'),'prompt_version':pv,'soft_prompt_version':SPV if shda else 'none','backfill_version':BV if shda else 'none','breaker_version':CKV if shda else 'none','notes':note,**rr});mets.append({'exp_id':eid,'system_name':name,'budget':b,'metrics':o['summary'],'reliability_metrics':rr});brl+=[{'run_id':rid,'exp_id':eid,'sample_id':x.get('sample_id',''),'budget':b,'breaker_status':x.get('breaker_status','not_triggered')} for x in ps if shda];JL(rd/f"{eid}_{'BestRouterOnly' if eid=='M5' else 'SHDApp'}_online_budget_{int(round(b*100))}_results.jsonl",ps)
    relrows=[{'run_id':rid,'exp_id':s['exp_id'],'system_name':s['system_name'],'budget':s['budget'],'num_samples':len(by[(s['exp_id'],s['budget'])]),**R(by[(s['exp_id'],s['budget'])],s['system_name']=='SH-DA++')} for s in sums];fails=[{'run_id':rid,'exp_id':x['exp_id'],'system_name':x['system_name'],'budget':x['budget'],'sample_id':x['sample_id'],'domain':x['domain'],'error_type':x['error_type'],'gt':x['gt'],'ocr_text':x['ocr_text'],'final_text':x['final_text'],'selected_for_upgrade':x['selected_for_upgrade'],'router_score':x['router_score'],'backfill_status':x['backfill_status'],'backfill_reason':x['backfill_reason'],'breaker_status':x.get('breaker_status','')} for x in allr if (x.get('is_correct_ocr') is False and x.get('is_correct_final') is False)];buds=[{'run_id':rid,'exp_id':s['exp_id'],'system_name':s['system_name'],'budget':s['budget'],'target_call_rate':s['target_call_rate'],'actual_call_rate':s['actual_call_rate'],'abs_diff':round(abs(float(s['actual_call_rate'])-float(s['target_call_rate'])),4),'valid_threshold':TOL,'call_rate_valid':'valid' if abs(float(s['actual_call_rate'])-float(s['target_call_rate']))<=TOL else 'invalid'} for s in sums]
    WC(rd/'summary.csv',list(sums[0].keys()),sums);WC(rd/'reliability_breakdown.csv',list(relrows[0].keys()),relrows);WC(rd/'failure_cases.csv',list(fails[0].keys()) if fails else ['run_id'],fails);WC(rd/'budget_check.csv',list(buds[0].keys()),buds);WC(rd/'domain_breakdown.csv',['run_id'],[]);J(rd/'metrics_summary.json',{'run_id':rid,'reuse_cached_vlm':a.reuse_cached_vlm,'m5_cache_path':str(m5p),'m6_cache_path':str(m6p),'results':mets});JL(rd/'backfill_log.jsonl',bfl);JL(rd/'breaker_log.jsonl',brl);J(rd/'runtime_stats.json',{'run_id':rid,'num_samples':len(rows),'budgets':a.budgets,'systems':['M5','M6'],'reuse_cached_vlm':a.reuse_cached_vlm});(rd/'run_manifest.md').write_text(f'run_id: {rid}\nbest_router_name: {BR}\nbudgets: {a.budgets}\nreuse_cached_vlm: {a.reuse_cached_vlm}\nstatus: done\n',encoding='utf-8');(rd/'env_snapshot.txt').write_text(f'python_version={sys.version.split()[0]}\ngit_commit={G(root)}\n',encoding='utf-8');print(rd)
