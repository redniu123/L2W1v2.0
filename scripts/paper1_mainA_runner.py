#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper1 Main A RouterOnly replay runner."""
import argparse,csv,json,random,sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import Levenshtein,numpy as np,yaml
from tqdm import tqdm
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from scripts.run_efficiency_frontier import build_agent_b_callable,ensure_agent_a_result_schema,infer_all_samples,summarize_extended_metrics,summarize_latency_and_token_usage
WMW,WNW,WDW=0.5,0.3,0.2;MTH,DTH,GB=0.35,0.20,0.10
STRATS=['GCR','WUR','DGCR','DWUR'];BUDGETS=[0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.80,1.00];MAIN={0.10,0.20,0.30}

def norm(t):
    return '' if not t else t.translate(str.maketrans({'（':'(','）':')','【':'[','】':']','｛':'{','｝':'}','，':',','：':':','；':';','！':'!','？':'?','。':'.'}))

def score(s,r,eta=0.5):
    mc=float(r.get('mean_conf',r.get('conf',0.0)));mi=float(r.get('min_conf',r.get('conf',0.0)));dr=float(r.get('drop',0.0));cf=float(r.get('conf',mc));rd=float(r.get('r_d',0.0))
    if s=='GCR': return 1.0-cf
    if s=='DGCR': return (1.0-cf)+rd
    q=WMW*(1.0-mc)+WNW*(1.0-mi)+WDW*dr
    if mi<MTH: q+=GB
    if dr>DTH: q+=GB
    return float(q if s=='WUR' else q+eta*rd)

def wjsonl(p,rows):
    with Path(p).open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')

def wcsv(p,fs,rows):
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fs);w.writeheader();[w.writerow({k:r.get(k,'') for k in fs}) for r in rows]

def build_full(agent_rows,agent_b,pv,ab,run_id):
    out=[]
    for r in tqdm(agent_rows,desc='Shared full-call cache'):
        rs=agent_b({'T_A':r['T_A'],'image_path':r.get('img_path',r.get('image_path','')),'min_conf_idx':r.get('min_conf_idx',-1),'sample_id':r.get('sample_id','')})
        up=rs.get('corrected_text',r['T_A']);up=up if isinstance(up,str) and up else r['T_A']
        out.append({'sample_id':r.get('sample_id',''),'image_path':r.get('image_path',''),'source_image_id':r.get('source_image_id',''),'domain':r.get('domain','geology'),'split':r.get('split','test'),'gt':r['T_GT'],'ocr_text':r['T_A'],'final_text_if_upgraded':up,'final_text':up,'vlm_raw_output':up,'latency_ms':rs.get('latency_ms'),'token_usage':rs.get('token_usage'),'error_type':rs.get('error_type','none'),'has_professional_terms':r.get('has_professional_terms',False),'professional_terms':r.get('professional_terms',[]),'domain_risk_score':round(float(r.get('r_d',0.0)),6),'mean_conf':r.get('mean_conf'),'min_conf':r.get('min_conf'),'drop':r.get('drop'),'conf':r.get('conf'),'r_d':r.get('r_d',0.0),'is_correct_ocr':r['T_A']==r['T_GT'],'edit_distance_ocr':Levenshtein.distance(norm(r['T_A']),norm(r['T_GT'])),'vlm_model':ab,'prompt_version':pv,'run_id':run_id})
    return out

def replay(strategy,budget,full,score_map,pv,run_id):
    ranked=sorted(range(len(full)),key=lambda i:score_map[i],reverse=True);n=int(round(len(full)*budget));up=set(ranked[:n]);rmap={i:k+1 for k,i in enumerate(ranked)}
    ps=[];cer=gtl=nup=nacc=0
    for i,it in enumerate(full):
        ta,tg=it['ocr_text'],it['gt'];sel=i in up;ft=it['final_text_if_upgraded'] if sel else ta
        if sel: nup+=1; nacc+=1 if ft!=ta else 0
        cer+=Levenshtein.distance(norm(ft),norm(tg)); gtl+=len(norm(tg))
        row=dict(it); row.update({'router_name':strategy,'router_score':round(score_map[i],6),'budget':budget,'budget_mode':'offline_replay','selected_for_upgrade':sel,'replay_rank':rmap[i],'final_text':ft,'vlm_raw_output':it['vlm_raw_output'] if sel else '','latency_ms':it['latency_ms'] if sel else None,'token_usage':it['token_usage'] if sel else None,'error_type':it['error_type'] if sel else 'not_upgraded','backfill_status':'skipped' if sel else 'not_upgraded','backfill_reason':'paper1_routeronly' if sel else 'not_upgraded','cvr_flag':False,'is_correct_final':ft==tg,'edit_distance_final':Levenshtein.distance(norm(ft),norm(tg)),'prompt_version':pv,'run_id':run_id}); ps.append(row)
    ex=summarize_extended_metrics(ps); us=summarize_latency_and_token_usage(ps); ar=(nup/len(full)) if full else 0.0
    return {'summary':{'run_id':run_id,'router_name':strategy,'budget':budget,'target_call_rate':budget,'actual_call_rate':round(ar,4),'call_rate_valid':abs(ar-budget)<=0.005,'CER':round(cer/gtl,6) if gtl else 0.0,'BoundaryDeletionRecallAtB':ex['Boundary_Deletion_Recall@B'],'SubstitutionCER':ex['Substitution_CER'],'AER':round(nacc/nup,4) if nup else 0.0,'CVR':0.0,'p95_latency_ms':us['P95_Latency_MS'],'avg_token_usage':us['Avg_Token_Usage'],'agentB_model':full[0]['vlm_model'] if full else '','prompt_version':pv,'n_valid':len(full)},'per_sample':ps}

def domain_rows(rows):
    acc=defaultdict(lambda:{'n':0,'c':0,'cer':0,'gt':0,'u':0})
    for r in rows:
        k=(r['router_name'],r['budget'],r['domain']);a=acc[k];a['n']+=1;a['u']+=1 if r['selected_for_upgrade'] else 0;a['c']+=1 if r['is_correct_final'] else 0;a['cer']+=Levenshtein.distance(norm(r['final_text']),norm(r['gt']));a['gt']+=len(norm(r['gt']))
    out=[]
    for (rn,b,d),a in sorted(acc.items()): out.append({'router_name':rn,'budget':b,'domain':d,'n_samples':a['n'],'actual_call_rate':round(a['u']/a['n'],4) if a['n'] else 0.0,'CER':round(a['cer']/a['gt'],6) if a['gt'] else 0.0,'final_accuracy':round(a['c']/a['n'],6) if a['n'] else 0.0})
    return out

def case_pool(rows):
    out=[]
    for r in rows:
        eo,ef=int(r.get('edit_distance_ocr',0)),int(r.get('edit_distance_final',0))
        if r.get('selected_for_upgrade') and r.get('final_text')!=r.get('ocr_text'):
            lab='beneficial' if ef<eo else 'harmful' if ef>eo else 'neutral'
            out.append({'sample_id':r.get('sample_id',''),'router_name':r.get('router_name',''),'budget':r.get('budget',0.0),'domain':r.get('domain',''),'ocr_text':r.get('ocr_text',''),'vlm_raw_output':r.get('vlm_raw_output',''),'final_text':r.get('final_text',''),'gt':r.get('gt',''),'edit_distance_ocr':eo,'edit_distance_final':ef,'delta_edit_distance':ef-eo,'case_label':lab,'router_score':r.get('router_score',0.0),'replay_rank':r.get('replay_rank')})
    return out

def main():
    p=argparse.ArgumentParser(description='paper1 Main A RouterOnly replay runner');p.add_argument('--config',default='configs/router_config.yaml');p.add_argument('--test_jsonl',default='data/l2w1data/test.jsonl');p.add_argument('--image_root',default='data/l2w1data/images');p.add_argument('--rec_model_dir',default='./models/agent_a_ppocr/PP-OCRv5_server_rec_infer');p.add_argument('--rec_char_dict_path',default='ppocr/utils/ppocrv5_dict.txt');p.add_argument('--geo_dict',default='data/dicts/Geology.txt');p.add_argument('--finance_dict',default='data/dicts/Finance.txt');p.add_argument('--medicine_dict',default='data/dicts/Medicine.txt');p.add_argument('--output_dir',default='paper1_runs/mainA');p.add_argument('--budgets',default=','.join(f'{b:.2f}' for b in BUDGETS));p.add_argument('--seed',type=int,default=42);p.add_argument('--n_samples',type=int,default=None);p.add_argument('--use_gpu',action='store_true',default=False);p.add_argument('--use_cache',action='store_true',default=False);p.add_argument('--rebuild_cache',action='store_true',default=False);p.add_argument('--prompt_version',default=None);p.add_argument('--agent_b_skip',action='store_true',default=False);a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);cfg=yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    if a.agent_b_skip: cfg=dict(cfg); cfg['agent_b']=dict(cfg.get('agent_b',{})); cfg['agent_b']['skip']=True
    pv=a.prompt_version or cfg.get('prompt_version') or cfg.get('mainline',{}).get('prompt_version','prompt_v1.1');ab=cfg.get('mainline_agent_b') or cfg.get('mainline',{}).get('mainline_agent_b','configured_agent_b');eta=float((cfg or {}).get('sh_da_v4',{}).get('rule_scorer',{}).get('eta',0.5));budgets=[float(x) for x in a.budgets.split(',') if x.strip()]
    import argparse as _ap
    rec=_ap.Namespace(rec_model_dir=a.rec_model_dir,rec_char_dict_path=a.rec_char_dict_path,rec_image_shape='3, 48, 320',rec_batch_num=6,rec_algorithm='SVTR_LCNet',use_space_char=True,use_gpu=a.use_gpu,use_xpu=False,use_npu=False,use_mlu=False,use_metax_gpu=False,use_gcu=False,ir_optim=True,use_tensorrt=False,min_subgraph_size=15,precision='fp32',gpu_mem=500,gpu_id=0,enable_mkldnn=None,cpu_threads=10,warmup=False,benchmark=False,save_log_path='./log_output/',show_log=False,use_onnx=False,max_batch_size=10,return_word_box=False,drop_score=0.5,max_text_length=25,rec_image_inverse=True,use_det=False,det_model_dir='')
    from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits
    from modules.router.domain_knowledge import DomainKnowledgeEngine
    recog=TextRecognizerWithLogits(rec); de=DomainKnowledgeEngine({'geology':a.geo_dict,'finance':a.finance_dict,'medicine':a.medicine_dict}); agent_b=build_agent_b_callable(cfg)
    samples=[json.loads(x) for x in Path(a.test_jsonl).read_text(encoding='utf-8').splitlines() if x.strip()]; out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); cache=out/'shared_agent_a_cache.json'
    if a.use_cache and not a.rebuild_cache and cache.exists(): ar=ensure_agent_a_result_schema(json.loads(cache.read_text(encoding='utf-8')))
    else:
        ar=ensure_agent_a_result_schema(infer_all_samples(samples,recog,de,None,a.image_root)); cache.write_text(json.dumps(ar,ensure_ascii=False),encoding='utf-8')
    if a.n_samples and a.n_samples<len(ar): ar=ar[:a.n_samples]
    run_id=datetime.now().strftime('%Y%m%d_run%H%M%S'); run_dir=out/run_id; run_dir.mkdir(parents=True,exist_ok=True)
    full=build_full(ar,agent_b,pv,ab,run_id); wjsonl(run_dir/'shared_repmodel_full_call_cache.jsonl',full); (run_dir/'config_snapshot.yaml').write_text(yaml.safe_dump({'run_id':run_id,'args':vars(a),'config':cfg},allow_unicode=True,sort_keys=False),encoding='utf-8')
    rs_rows=[]; rs_map={s:[] for s in STRATS}
    for r in ar:
        rr={'sample_id':r.get('sample_id',''),'domain':r.get('domain','geology')}
        for s in STRATS: sc=score(s,r,eta=eta); rs_map[s].append(sc); rr[f'{s}_score']=round(sc,6)
        rs_rows.append(rr)
    wcsv(run_dir/'router_score_matrix.csv',list(rs_rows[0].keys()) if rs_rows else ['sample_id'],rs_rows)
    main_rows=[]; all_ps=[]
    for s in STRATS:
        for b in budgets:
            res=replay(s,b,full,rs_map[s],pv,run_id); main_rows.append(res['summary']); all_ps.extend(res['per_sample']); wjsonl(run_dir/f"offline_budget_{int(round(b*100)):02d}_{s}.jsonl",res['per_sample'])
    wcsv(run_dir/'tab_mainA_results.csv',list(main_rows[0].keys()) if main_rows else ['router_name'],main_rows)
    dr=domain_rows(all_ps); wcsv(run_dir/'tab_mainA_domain_results.csv',list(dr[0].keys()) if dr else ['router_name'],dr)
    br=[r for r in main_rows if round(float(r['budget']),2) in MAIN]; wcsv(run_dir/'tab_mainA_budget_check.csv',list(br[0].keys()) if br else ['router_name'],br)
    cp=case_pool(all_ps); wcsv(run_dir/'mainA_case_pool.csv',list(cp[0].keys()) if cp else ['sample_id'],cp)
    print(run_dir)
if __name__=='__main__': main()
