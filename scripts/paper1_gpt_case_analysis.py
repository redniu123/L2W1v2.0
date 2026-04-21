#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse,csv,json,re
from collections import Counter
from datetime import datetime
from pathlib import Path

STRUCT_CHARS='()[]{}<>〈〉《》【】（）{}→←↳↔|:：;；,.。·+-—~～°'
NUM_PAT=r'[0-9０-９ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVXivx]'

def rjsonl(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def wcsv(p,rows):
    p=Path(p)
    if not rows: p.write_text('',encoding='utf-8'); return
    fields=[]; seen=set()
    for row in rows:
        for k in row.keys():
            if k not in seen: seen.add(k); fields.append(k)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def bucket(row):
    gt,ocr,fin=row['gt'],row['ocr_text'],row['final_text']; eo=int(row.get('edit_distance_ocr',0)); ef=int(row.get('edit_distance_final',0))
    if ef<=eo: return 'not_harmful'
    if (any(c in (gt+ocr) for c in STRUCT_CHARS) and any(c in (fin+gt) for c in STRUCT_CHARS)) or (sum(c in STRUCT_CHARS for c in fin)!=sum(c in STRUCT_CHARS for c in ocr)): return 'structure_symbol_break'
    if re.search(NUM_PAT,gt+ocr+fin): return 'number_id_break'
    if row.get('professional_terms'): return 'term_entity_mischange'
    if len(fin)!=len(ocr) or sum(a!=b for a,b in zip(fin,ocr))>=2: return 'semantic_overrewrite'
    return 'other'
def load_model_rows(run_dir,prefix,budget_tag): return {r['sample_id']:r for r in rjsonl(Path(run_dir)/f'{prefix}_offline_budget_{budget_tag}.jsonl')}
def pick_top(rows,k,cond,key): return sorted([r for r in rows if cond(r)],key=key,reverse=True)[:k]
def md_block(title,rows):
    out=[f'## {title}','','|sample_id|domain|ocr_text|final_text|gt|note|','|---|---|---|---|---|---|']
    for r in rows:
        note=r.get('case_note','').replace('|','/'); out.append(f"|{r.get('sample_id','')}|{r.get('domain','')}|{r.get('ocr_text','').replace('|','/')}|{r.get('final_text','').replace('|','/')}|{r.get('gt','').replace('|','/')}|{note}|")
    out.append(''); return '\n'.join(out)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--maina_run_dir',default='paper1_runs/mainA/20260417_run112203'); p.add_argument('--mainc_run_dir',default='paper1_runs/mainC/20260421_run050030'); p.add_argument('--output_root',default='paper1_runs/phase2_batch2'); p.add_argument('--budget_tag',default='30'); a=p.parse_args()
    ad=Path(a.maina_run_dir); cd=Path(a.mainc_run_dir); out=Path(a.output_root)/datetime.now().strftime('%Y%m%d_run%H%M%S'); out.mkdir(parents=True,exist_ok=True)
    gpt=load_model_rows(cd,'V4',a.budget_tag); gem=load_model_rows(cd,'V3',a.budget_tag); qwen=load_model_rows(cd,'V1',a.budget_tag)
    all_gpt=list(gpt.values()); harmful=[r for r in all_gpt if r.get('selected_for_upgrade') and int(r.get('edit_distance_final',0))>int(r.get('edit_distance_ocr',0))]; good=[r for r in all_gpt if r.get('selected_for_upgrade') and int(r.get('edit_distance_final',0))<int(r.get('edit_distance_ocr',0))]
    cnt=Counter(bucket(r) for r in harmful); total=max(1,len(harmful)); wcsv(out/'gpt_error_bucket_stats.csv',[{'model_name':'gpt-5.4','bucket':k,'count':v,'ratio':round(v/total,6)} for k,v in cnt.most_common()])
    style=[]
    for name,data in [('gpt-5.4',gpt),('Gemini 3 Flash Preview',gem),('Qwen3-VL-8B',qwen)]:
        rows=list(data.values()); up=[r for r in rows if r.get('selected_for_upgrade')]; harm=[r for r in up if int(r.get('edit_distance_final',0))>int(r.get('edit_distance_ocr',0))]; ben=[r for r in up if int(r.get('edit_distance_final',0))<int(r.get('edit_distance_ocr',0))]; c=Counter(bucket(r) for r in harm)
        style.append({'model_name':name,'n_upgraded':len(up),'n_harmful':len(harm),'n_beneficial':len(ben),'harmful_ratio':round(len(harm)/len(up),6) if up else 0.0,'beneficial_ratio':round(len(ben)/len(up),6) if up else 0.0,'structure_symbol_break':c.get('structure_symbol_break',0),'number_id_break':c.get('number_id_break',0),'term_entity_mischange':c.get('term_entity_mischange',0),'semantic_overrewrite':c.get('semantic_overrewrite',0),'other':c.get('other',0)})
    wcsv(out/'model_error_style_comparison.csv',style)
    maina=[]
    casep=ad/'mainA_case_pool.csv'
    if casep.exists():
        with casep.open('r',encoding='utf-8') as f: maina=list(csv.DictReader(f))
    gcr_success=pick_top(maina,12,lambda r: r.get('router_name')=='GCR' and str(r.get('budget'))=='0.3' and r.get('case_label')=='beneficial',lambda r: int(r.get('edit_distance_ocr',0))-int(r.get('edit_distance_final',0)))
    for r in gcr_success: r['case_group']='gcr_success'; r['case_note']='GCR 选中后误差下降'
    gpt_negative=pick_top(harmful,12,lambda r: True,lambda r: int(r.get('edit_distance_final',0))-int(r.get('edit_distance_ocr',0)))
    for r in gpt_negative: r['case_group']='gpt_negative'; r['case_note']=bucket(r)
    cross=[]
    for sid,gr in gem.items():
        qr=qwen.get(sid); vr=gpt.get(sid)
        if not qr or not vr or not gr.get('selected_for_upgrade'): continue
        eg,eq,ev,eo=int(gr.get('edit_distance_final',0)),int(qr.get('edit_distance_final',0)),int(vr.get('edit_distance_final',0)),int(gr.get('edit_distance_ocr',0))
        if eg<eo and eq>=eo and ev>=eo:
            d=dict(gr); d['qwen_final_text']=qr.get('final_text',''); d['gpt_final_text']=vr.get('final_text',''); d['case_group']='gemini_advantage'; d['case_note']='Gemini 修好，Qwen/GPT 未修好'; cross.append(d)
    cross=sorted(cross,key=lambda r: int(r.get('edit_distance_ocr',0))-int(r.get('edit_distance_final',0)),reverse=True)[:12]
    edge=[]
    for sid,gr in gem.items():
        qr=qwen.get(sid); vr=gpt.get(sid)
        if not qr or not vr: continue
        eg,eq,ev,eo=int(gr.get('edit_distance_final',0)),int(qr.get('edit_distance_final',0)),int(vr.get('edit_distance_final',0)),int(gr.get('edit_distance_ocr',0))
        if min(eg,eq,ev)<eo and max(eg,eq,ev)>eo:
            d=dict(gr); d['qwen_final_text']=qr.get('final_text',''); d['gpt_final_text']=vr.get('final_text',''); d['case_group']='edge_mixed'; d['case_note']='不同模型行为分化'; edge.append(d)
    edge=sorted(edge,key=lambda r: abs(int(r.get('edit_distance_ocr',0))-int(r.get('edit_distance_final',0))),reverse=True)[:12]
    casebook=gcr_success+gpt_negative+cross+edge; wcsv(out/'main_casebook.csv',casebook)
    (out/'figure_case_samples.md').write_text('\n\n'.join([md_block('GCR 成功挑样案例',gcr_success),md_block('GPT 负优化典型案例',gpt_negative),md_block('Gemini 优于 Qwen/GPT 的案例',cross),md_block('边缘/分化案例',edge)]),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps({'maina_run_dir':str(ad),'mainc_run_dir':str(cd),'budget_tag':a.budget_tag,'n_gpt_harmful':len(harmful),'n_gpt_beneficial':len(good)},ensure_ascii=False,indent=2),encoding='utf-8'); print(out)
if __name__=='__main__': main()
