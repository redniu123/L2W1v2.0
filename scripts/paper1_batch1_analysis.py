#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse,csv,json,random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import Levenshtein

def norm(t): return '' if not t else t.translate(str.maketrans({'（':'(','）':')','【':'[','】':']','｛':'{','｝':'}','，':',','：':':','；':';','！':'!','？':'?','。':'.'}))
def rjsonl(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def rcsv(p):
    with Path(p).open('r',encoding='utf-8') as f: return list(csv.DictReader(f))
def wcsv(p,rows):
    p=Path(p)
    if not rows: p.write_text('',encoding='utf-8'); return
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
def cer_full(rows):
    num=sum(int(r.get('edit_distance_ocr',0)) for r in rows); den=sum(len(norm(r.get('gt',''))) for r in rows); return (num/den) if den else 0.0
def summarize(ps,run_id,label,b,pv,n):
    cer=sum(Levenshtein.distance(norm(r['final_text']),norm(r['gt'])) for r in ps); gtl=sum(len(norm(r['gt'])) for r in ps); nu=sum(1 for r in ps if r['selected_for_upgrade']); na=sum(1 for r in ps if r['selected_for_upgrade'] and r['final_text']!=r['ocr_text']); lat=sorted(float(r['latency_ms']) for r in ps if r.get('latency_ms') is not None); p95=lat[min(len(lat)-1,int(0.95*(len(lat)-1)))] if lat else 0.0
    return {'run_id':run_id,'router_name':label,'budget':b,'target_call_rate':b,'actual_call_rate':round(nu/n,4) if n else 0.0,'call_rate_valid':abs((nu/n)-b)<=0.005 if n else False,'CER':round(cer/gtl,6) if gtl else 0.0,'AER':round(na/nu,4) if nu else 0.0,'p95_latency_ms':round(p95,3),'agentB_model':'gemini-3-flash-preview','prompt_version':pv,'n_valid':n}
def replay(full,order,b,run_id,label):
    n=len(full); k=int(round(n*b)); up=set(order[:k]); rmap={idx:i+1 for i,idx in enumerate(order)}; out=[]
    for i,r in enumerate(full):
        ocr=r['ocr_text']; gt=r['gt']; sel=i in up; ft=r['final_text_if_upgraded'] if sel else ocr; d=dict(r)
        d.update({'router_name':label,'budget':b,'selected_for_upgrade':sel,'replay_rank':rmap[i],'final_text':ft,'vlm_raw_output':r['vlm_raw_output'] if sel else '','latency_ms':r.get('latency_ms') if sel else None,'token_usage':r.get('token_usage') if sel else None,'error_type':r.get('error_type','none') if sel else 'not_upgraded','is_correct_final':ft==gt,'edit_distance_final':Levenshtein.distance(norm(ft),norm(gt)),'run_id':run_id})
        out.append(d)
    return out
def case_pool(rows):
    out=[]
    for r in rows:
        if not r.get('selected_for_upgrade'): continue
        eo=int(r.get('edit_distance_ocr',0)); ef=int(r.get('edit_distance_final',0)); lab='beneficial' if ef<eo else 'harmful' if ef>eo else 'neutral'
        out.append({'sample_id':r.get('sample_id',''),'router_name':r.get('router_name',''),'budget':r.get('budget',0.0),'domain':r.get('domain',''),'ocr_text':r.get('ocr_text',''),'final_text':r.get('final_text',''),'gt':r.get('gt',''),'edit_distance_ocr':eo,'edit_distance_final':ef,'delta_edit_distance':ef-eo,'case_label':lab,'replay_rank':r.get('replay_rank')})
    return out
def agg_domain(rows,keys):
    acc=defaultdict(lambda:{'n':0,'u':0,'c':0,'cer':0,'gt':0})
    for r in rows:
        k=tuple(r[x] for x in keys); a=acc[k]; a['n']+=1; a['u']+=1 if r.get('selected_for_upgrade') else 0; a['c']+=1 if r.get('is_correct_final') else 0; a['cer']+=int(r.get('edit_distance_final',0)); a['gt']+=len(norm(r.get('gt','')))
    out=[]
    for k,a in sorted(acc.items()):
        d={x:v for x,v in zip(keys,k)}; d.update({'n_samples':a['n'],'actual_call_rate':round(a['u']/a['n'],4) if a['n'] else 0.0,'CER':round(a['cer']/a['gt'],6) if a['gt'] else 0.0,'final_accuracy':round(a['c']/a['n'],6) if a['n'] else 0.0}); out.append(d)
    return out
def router_rank(rows):
    g=defaultdict(list)
    for r in rows: g[(r['domain'],r['budget'])].append(r)
    out=[]
    for (d,b),rs in sorted(g.items()):
        rs=sorted(rs,key=lambda x: float(x['CER']))
        for i,r in enumerate(rs,1): out.append({'domain':d,'budget':b,'router_name':r['router_name'],'rank_by_cer':i,'CER':r['CER'],'final_accuracy':r['final_accuracy']})
    return out
def cost_rows(maina,mainc,base):
    out=[]
    for src,rows in (('mainA',maina),('mainC',mainc)):
        for r in rows:
            cer=float(r['CER']); budget=float(r.get('actual_call_rate',r['budget'])); absr=max(0.0,base-cer); rel=(absr/base) if base else 0.0
            out.append({'source':src,'name':r.get('router_name') or r.get('model_name'),'budget':r['budget'],'actual_call_rate':r.get('actual_call_rate',r['budget']),'CER':r['CER'],'base_ocr_cer':round(base,6),'absolute_cer_reduction':round(absr,6),'relative_error_reduction':round(rel,6),'Errors_Reduced_per_API_Call':round(absr/budget,6) if budget else 0.0,'Delta_CER_per_Budget':round(absr/budget,6) if budget else 0.0,'Relative_Error_Reduction_per_Budget':round(rel/budget,6) if budget else 0.0})
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--maina_run_dir',default='paper1_runs/mainA/20260417_run112203'); p.add_argument('--mainc_run_dir',default='paper1_runs/mainC/20260421_run050030'); p.add_argument('--output_root',default='paper1_runs/phase2_batch1'); p.add_argument('--random_seeds',default='1,2,3,4,5'); a=p.parse_args()
    ad=Path(a.maina_run_dir); cd=Path(a.mainc_run_dir); out=Path(a.output_root)/datetime.now().strftime('%Y%m%d_run%H%M%S'); out.mkdir(parents=True,exist_ok=True)
    full=rjsonl(ad/'shared_repmodel_full_call_cache.jsonl'); maina=rcsv(ad/'tab_mainA_results.csv'); mainc=rcsv(cd/'tab_mainC_results.csv'); budgets=[float(r['budget']) for r in maina if r['router_name']=='GCR']; pv=full[0].get('prompt_version','prompt_v1.1') if full else 'prompt_v1.1'; run_id=out.name; base=cer_full(full)
    order=sorted(range(len(full)),key=lambda i:(1.0-float(full[i].get('min_conf',1.0)),1.0-float(full[i].get('conf',1.0))),reverse=True); msum=[]; mall=[]
    for b in budgets:
        ps=replay(full,order,b,run_id,'MinConf'); mall.extend(ps); msum.append(summarize(ps,run_id,'MinConf',b,pv,len(full)))
    wcsv(out/'tab_minconf_baseline_results.csv',msum); wcsv(out/'minconf_case_pool.csv',case_pool(mall))
    seeds=[int(x) for x in a.random_seeds.split(',') if x.strip()]; rs=[]; rall=[]
    for seed in seeds:
        order=list(range(len(full))); random.Random(seed).shuffle(order)
        for b in budgets:
            ps=replay(full,order,b,run_id,f'Random_s{seed}'); rall.extend(ps); s=summarize(ps,run_id,f'Random_s{seed}',b,pv,len(full)); s['seed']=seed; rs.append(s)
    wcsv(out/'tab_random_seed_results.csv',rs)
    grp=defaultdict(list)
    for r in rs: grp[float(r['budget'])].append(r)
    mean_rows=[]
    for b,rows in sorted(grp.items()):
        c=[float(r['CER']) for r in rows]; aers=[float(r['AER']) for r in rows]; calls=[float(r['actual_call_rate']) for r in rows]; cm=sum(c)/len(c)
        mean_rows.append({'run_id':run_id,'router_name':'Random','budget':b,'target_call_rate':b,'actual_call_rate_mean':round(sum(calls)/len(calls),4),'CER_mean':round(cm,6),'CER_std':round((sum((x-cm)**2 for x in c)/len(c))**0.5,6),'AER_mean':round(sum(aers)/len(aers),4),'n_seeds':len(rows)})
    wcsv(out/'tab_random_baseline_results.csv',mean_rows); wcsv(out/'random_case_pool.csv',case_pool(rall))
    maina_domain=[]
    for pth in sorted(ad.glob('offline_budget_*_*.jsonl')): maina_domain.extend(agg_domain(rjsonl(pth),['router_name','budget','domain']))
    wcsv(out/'tab_domain_budget_curve.csv',[r for r in maina_domain if r['router_name']=='GCR']); wcsv(out/'tab_domain_router_ranking.csv',router_rank(maina_domain))
    model_domain=[]
    for pth in sorted(cd.glob('V*_offline_budget_*.jsonl')): model_domain.extend(agg_domain(rjsonl(pth),['model_name','budget','domain']))
    wcsv(out/'tab_domain_model_comparison.csv',model_domain)
    wcsv(out/'tab_cost_effectiveness.csv',cost_rows(maina,mainc,base))
    (out/'manifest.json').write_text(json.dumps({'maina_run_dir':str(ad),'mainc_run_dir':str(cd),'base_ocr_cer':round(base,6),'random_seeds':seeds},ensure_ascii=False,indent=2),encoding='utf-8'); print(out)
if __name__=='__main__': main()
