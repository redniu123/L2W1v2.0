#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper1 Main C cross-model RouterOnly runner."""
import argparse,csv,json,random,sys,time
from datetime import datetime
from pathlib import Path
import Levenshtein,numpy as np,requests,yaml
from tqdm import tqdm
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from scripts.run_efficiency_frontier import ensure_agent_a_result_schema,infer_all_samples,summarize_extended_metrics,summarize_latency_and_token_usage,build_agent_b_callable
from modules.vlm_expert import AgentBFactory
from modules.vlm_expert.provider_pools import get_provider_pool
M={'V1':('Qwen3-VL-8B','local_vlm','qwen2.5_vl','./models/agent_b_vlm/Qwen3-VL-8B'),'V2':('InternVL3-8B','local_vlm','internvl2_5','./models/agent_b_vlm/InternVL3-8B'),'V3':('Gemini 3 Flash Preview','gemini','',''),'V4':('Claude Sonnet 4.6','claude','','')}
B=[0.05,0.10,0.20,0.30,0.50,0.80,1.00]
WMW,WNW,WDW=0.5,0.3,0.2;MTH,DTH,GB=0.35,0.20,0.10

def norm(t): return '' if not t else t.translate(str.maketrans({'（':'(','）':')','【':'[','】':']','｛':'{','｝':'}','，':',','：':':','；':';','！':'!','？':'?','。':'.'}))
def wjsonl(p,rows):
    with Path(p).open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
def wcsv(p,fs,rows):
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fs);w.writeheader();[w.writerow({k:r.get(k,'') for k in fs}) for r in rows]
def s(rt,r,eta):
    mc=float(r.get('mean_conf',r.get('conf',0.0)));mi=float(r.get('min_conf',r.get('conf',0.0)));dr=float(r.get('drop',0.0));cf=float(r.get('conf',mc));rd=float(r.get('r_d',0.0))
    if rt=='GCR': return 1.0-cf
    if rt=='DGCR': return (1.0-cf)+rd
    q=WMW*(1.0-mc)+WNW*(1.0-mi)+WDW*dr
    if mi<MTH: q+=GB
    if dr>DTH: q+=GB
    return float(q if rt=='WUR' else q+eta*rd)
def claude(cfg):
    pool=get_provider_pool('claude_sonnet_46',cfg['agent_b'].get('key_file','key.txt'));base_url=pool.base_url;api_key=pool.keys[0];model_name=pool.model_name;timeout=cfg['agent_b'].get('timeout',360)
    def call(p):
        t0=time.perf_counter();ocr=p.get('T_A','');payload={'model':model_name,'messages':[{'role':'user','content':[{'type':'text','text':f'你是中文OCR纠错助手。仅输出修正后的完整文本。原始OCR文本：{ocr}'}]}],'temperature':0.1,'max_tokens':256}
        try:
            r=requests.post(f'{base_url}/chat/completions',headers={'Authorization':f'Bearer {api_key}','Content-Type':'application/json'},json=payload,timeout=timeout);r.raise_for_status();txt=r.json()['choices'][0]['message']['content'].strip().split('\n')[0].strip();return {'corrected_text':txt or ocr,'latency_ms':round((time.perf_counter()-t0)*1000,3),'token_usage':None,'error_type':'none'}
        except Exception as e: return {'corrected_text':ocr,'latency_ms':round((time.perf_counter()-t0)*1000,3),'token_usage':None,'error_type':type(e).__name__}
    return call
def mk(backend,model_type,model_path,cfg,skip):
    if skip: return lambda p:{'corrected_text':p.get('T_A',''),'latency_ms':0.0,'token_usage':None,'error_type':'mock_skip'}
    if backend=='gemini': return build_agent_b_callable(cfg)
    if backend=='claude': return claude(cfg)
    cc=json.loads(json.dumps(cfg));cc['agent_b']={'skip':False,'backend':'local_vlm','model_type':model_type,'model_path':model_path,'torch_dtype':'float16','max_new_tokens':128};expert=AgentBFactory.create(cc)
    def call(p):
        t0=time.perf_counter();ta=p.get('T_A','');img=p.get('image_path') or p.get('img_path','');i=p.get('min_conf_idx',-1) or -1;sc=ta[i] if 0<=i<len(ta) else ''
        try:
            r=expert.process_hard_sample(img,{'ocr_text':ta,'suspicious_index':i,'suspicious_char':sc,'risk_level':'medium'});corr=r.get('corrected_text',ta) if isinstance(r,dict) else (r if isinstance(r,str) else ta);tok=r.get('token_usage') if isinstance(r,dict) else None;err=r.get('error_type','none') if isinstance(r,dict) else 'none';return {'corrected_text':corr,'latency_ms':round((time.perf_counter()-t0)*1000,3),'token_usage':tok,'error_type':err}
        except Exception as e: return {'corrected_text':ta,'latency_ms':round((time.perf_counter()-t0)*1000,3),'token_usage':None,'error_type':type(e).__name__}
    return call
def fc(rows,agent_b,name,pv,run_id):
    out=[]
    for r in tqdm(rows,desc=f'Full-call {name}',leave=False):
        rs=agent_b({'T_A':r['T_A'],'image_path':r.get('img_path',r.get('image_path','')),'min_conf_idx':r.get('min_conf_idx',-1),'sample_id':r.get('sample_id','')});up=rs.get('corrected_text',r['T_A']);up=up if isinstance(up,str) and up else r['T_A'];out.append({'sample_id':r.get('sample_id',''),'image_path':r.get('image_path',''),'source_image_id':r.get('source_image_id',''),'domain':r.get('domain','geology'),'split':r.get('split','test'),'gt':r['T_GT'],'ocr_text':r['T_A'],'final_text_if_upgraded':up,'vlm_raw_output':up,'latency_ms':rs.get('latency_ms'),'token_usage':rs.get('token_usage'),'error_type':rs.get('error_type','none'),'has_professional_terms':r.get('has_professional_terms',False),'professional_terms':r.get('professional_terms',[]),'is_correct_ocr':r['T_A']==r['T_GT'],'edit_distance_ocr':Levenshtein.distance(norm(r['T_A']),norm(r['T_GT'])),'vlm_model':name,'prompt_version':pv,'run_id':run_id})
    return out
def main():
    p=argparse.ArgumentParser(description='paper1 Main C cross-model RouterOnly runner');p.add_argument('--config',default='configs/router_config.yaml');p.add_argument('--test_jsonl',default='data/l2w1data/test.jsonl');p.add_argument('--image_root',default='data/l2w1data/images');p.add_argument('--rec_model_dir',default='./models/agent_a_ppocr/PP-OCRv5_server_rec_infer');p.add_argument('--rec_char_dict_path',default='ppocr/utils/ppocrv5_dict.txt');p.add_argument('--geo_dict',default='data/dicts/Geology.txt');p.add_argument('--finance_dict',default='data/dicts/Finance.txt');p.add_argument('--medicine_dict',default='data/dicts/Medicine.txt');p.add_argument('--output_dir',default='paper1_runs/mainC');p.add_argument('--best_router',default='WUR',choices=['GCR','WUR','DGCR','DWUR']);p.add_argument('--budgets',default=','.join(f'{b:.2f}' for b in B));p.add_argument('--models',default='V1,V2,V3,V4');p.add_argument('--seed',type=int,default=42);p.add_argument('--n_samples',type=int,default=None);p.add_argument('--use_gpu',action='store_true',default=False);p.add_argument('--use_cache',action='store_true',default=False);p.add_argument('--rebuild_cache',action='store_true',default=False);p.add_argument('--prompt_version',default=None);p.add_argument('--agent_b_skip',action='store_true',default=False);a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed)
    cfg=yaml.safe_load(Path(a.config).read_text(encoding='utf-8'));pv=a.prompt_version or cfg.get('prompt_version') or cfg.get('mainline',{}).get('prompt_version','prompt_v1.1');eta=float((cfg or {}).get('sh_da_v4',{}).get('rule_scorer',{}).get('eta',0.5));budgets=[float(x) for x in a.budgets.split(',') if x.strip()];chosen={x.strip() for x in a.models.split(',') if x.strip()}
    import argparse as _ap
    rec=_ap.Namespace(rec_model_dir=a.rec_model_dir,rec_char_dict_path=a.rec_char_dict_path,rec_image_shape='3, 48, 320',rec_batch_num=6,rec_algorithm='SVTR_LCNet',use_space_char=True,use_gpu=a.use_gpu,use_xpu=False,use_npu=False,use_mlu=False,use_metax_gpu=False,use_gcu=False,ir_optim=True,use_tensorrt=False,min_subgraph_size=15,precision='fp32',gpu_mem=500,gpu_id=0,enable_mkldnn=None,cpu_threads=10,warmup=False,benchmark=False,save_log_path='./log_output/',show_log=False,use_onnx=False,max_batch_size=10,return_word_box=False,drop_score=0.5,max_text_length=25,rec_image_inverse=True,use_det=False,det_model_dir='')
    from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits
    from modules.router.domain_knowledge import DomainKnowledgeEngine
    recog=TextRecognizerWithLogits(rec);de=DomainKnowledgeEngine({'geology':a.geo_dict,'finance':a.finance_dict,'medicine':a.medicine_dict});samples=[json.loads(x) for x in Path(a.test_jsonl).read_text(encoding='utf-8').splitlines() if x.strip()];out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);cache=out/'shared_agent_a_cache.json'
    if a.use_cache and not a.rebuild_cache and cache.exists(): rows=ensure_agent_a_result_schema(json.loads(cache.read_text(encoding='utf-8')))
    else: rows=ensure_agent_a_result_schema(infer_all_samples(samples,recog,de,None,a.image_root));cache.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
    if a.n_samples and a.n_samples<len(rows): rows=rows[:a.n_samples]
    ranked=sorted(range(len(rows)),key=lambda i:s(a.best_router,rows[i],eta),reverse=True);run_id=datetime.now().strftime('%Y%m%d_run%H%M%S');run_dir=out/run_id;run_dir.mkdir(parents=True,exist_ok=True)
    main_rows=[]
    for exp_id in chosen:
        if exp_id not in M: continue
        name,backend,model_type,model_path=M[exp_id];agent_b=mk(backend,model_type,model_path,cfg,a.agent_b_skip);full=fc(rows,agent_b,name,pv,run_id);wjsonl(run_dir/f'{exp_id}_full_call_cache.jsonl',full)
        for b in budgets:
            n=int(round(len(full)*b));up=set(ranked[:n]);rmap={i:k+1 for k,i in enumerate(ranked)};ps=[];cer=gtl=nup=nacc=0
            for i,it in enumerate(full):
                ta,tg=it['ocr_text'],it['gt'];sel=i in up;ft=it['final_text_if_upgraded'] if sel else ta
                if sel: nup+=1; nacc+=1 if ft!=ta else 0
                cer+=Levenshtein.distance(norm(ft),norm(tg));gtl+=len(norm(tg));row=dict(it);row.update({'model_name':name,'router_name':a.best_router,'budget':b,'target_call_rate':b,'selected_for_upgrade':sel,'replay_rank':rmap[i],'final_text':ft,'latency_ms':it['latency_ms'] if sel else None,'token_usage':it['token_usage'] if sel else None,'error_type':it['error_type'] if sel else 'not_upgraded','backfill_status':'skipped' if sel else 'not_upgraded','backfill_reason':'paper1_routeronly' if sel else 'not_upgraded','is_correct_final':ft==tg,'edit_distance_final':Levenshtein.distance(norm(ft),norm(tg))});ps.append(row)
            ex=summarize_extended_metrics(ps);us=summarize_latency_and_token_usage(ps);ar=(nup/len(full)) if full else 0.0;main_rows.append({'run_id':run_id,'model_name':name,'router_name':a.best_router,'budget':b,'target_call_rate':b,'actual_call_rate':round(ar,4),'call_rate_valid':abs(ar-b)<=0.005,'CER':round(cer/gtl,6) if gtl else 0.0,'BoundaryDeletionRecallAtB':ex['Boundary_Deletion_Recall@B'],'SubstitutionCER':ex['Substitution_CER'],'AER':round(nacc/nup,4) if nup else 0.0,'CVR':0.0,'p95_latency_ms':us['P95_Latency_MS'],'avg_token_usage':us['Avg_Token_Usage'],'prompt_version':pv,'n_valid':len(full)})
            wjsonl(run_dir/f'{exp_id}_offline_budget_{int(round(b*100)):02d}.jsonl',ps)
    wcsv(run_dir/'tab_mainC_results.csv',list(main_rows[0].keys()) if main_rows else ['model_name'],main_rows);wcsv(run_dir/'tab_mainC_budget_check.csv',list(main_rows[0].keys()) if main_rows else ['model_name'],[r for r in main_rows if round(float(r['budget']),2) in {0.10,0.20,0.30}]);print(run_dir)
if __name__=='__main__': main()
