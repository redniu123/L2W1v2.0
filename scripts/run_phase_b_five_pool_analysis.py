#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path

TERM_MIN_LEN=2;BOUNDARY_WINDOW=2;TOP_K_CASES=30
NUMBERING=set('0123456789①②③④⑤⑥⑦⑧⑨⑩ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ一二三四五六七八九十ABCDEF')
ROMAN=set('ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ①②③④⑤⑥⑦⑧⑨⑩一二三四五六七八九十')
ARABIC=set('0123456789ABCDEF')
STRUCT=set('()（）[]【】{}<>《》→—-=:：;；,.，。!?！？/\\|·•○●▲△※☆★~')

def jdump(p,o): Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def read_jsonl(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def wcsv(p,fs,rows):
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fs);w.writeheader()
        for r in rows:w.writerow({k:r.get(k,'') for k in fs})
def btag(b): return f'{int(round(b*100)):02d}'
def norm(t): return t.translate(str.maketrans({'（':'(','）':')','【':'[','】':']','｛':'{','｝':'}','，':',','：':':','；':';','！':'!','？':'?','。':'.','—':'-','–':'-','－':'-','→':'->','　':' '})).replace(' ','').strip()
def cset(t,a): return {c for c in t if c in a}
def terms(ts,txts): return [t for t in (ts or []) if len(t)>=TERM_MIN_LEN and any(t in x for x in txts)]
def dpos(a,b):
    l,m=0,min(len(a),len(b))
    while l<m and a[l]==b[l]: l+=1
    if l==len(a)==len(b): return None,None
    ra,rb=len(a)-1,len(b)-1
    while ra>=l and rb>=l and a[ra]==b[rb]: ra-=1;rb-=1
    return l,max(ra,rb)
def stype(a,b):
    if a==b:return 'no_change'
    l,r=dpos(a,b)
    if l is None:return 'no_change'
    if len(a)!=len(b):return 'length_changed'
    n=max(1,r-l+1)
    return 'single_char' if n==1 else 'local_multi_char' if n<=3 else 'long_span'
def is_boundary(a,b):
    if a==b:return False
    l,r=dpos(a,b)
    if l is None:return False
    n=max(len(a),len(b))
    return l<BOUNDARY_WINDOW or r>=n-BOUNDARY_WINDOW
def is_fmt(a,b): return a!=b and norm(a)==norm(b)
def is_sem(a,b,gt):
    if a==b or len(a)!=len(b): return False
    if cset(a+b,NUMBERING) or cset(a+b,STRUCT): return False
    return b!=gt and stype(a,b) in {'local_multi_char','long_span'}
def sem_sub(a,b,gt,ts):
    if not is_sem(a,b,gt): return 'none'
    if terms(ts,[a,b,gt]): return 'domain_term_shift'
    if b==gt: return 'semantic_fix'
    return 'near_synonym_or_shape_confusion' if len(a)==len(b) else 'semantic_hallucination'
def num_sub(a,b):
    if cset(a,NUMBERING)==cset(b,NUMBERING): return 'none'
    ar,br,aa,ba=bool(cset(a,ROMAN)),bool(cset(b,ROMAN)),bool(cset(a,ARABIC)),bool(cset(b,ARABIC))
    if ar and br and len(a)==len(b): return 'roman_numeral_fix'
    if aa and ba and len(a)==len(b): return 'arabic_digit_fix'
    if (ar or aa) and (br or ba): return 'enumeration_style_switch'
    return 'numbering_content_rewrite'
def bd_sub(a,b):
    if not is_boundary(a,b): return 'none'
    l,r=dpos(a,b);n=max(len(a),len(b));left,right=l<BOUNDARY_WINDOW,r>=n-BOUNDARY_WINDOW
    if cset(a,STRUCT)!=cset(b,STRUCT): return 'boundary_plus_structure_symbol'
    if left and right: return 'dual_boundary_edit'
    if stype(a,b)=='single_char': return 'left_boundary_single_char_fix' if left else 'right_boundary_single_char_fix'
    return 'left_boundary_multi_char_fix' if left else 'right_boundary_multi_char_fix'
def gain(o,v): return 'beneficial' if v<o else 'neutral' if v==o else 'harmful'
def action(r):
    if r['is_format_normalization'] or r['semantic_subclass'] in {'domain_term_shift','semantic_hallucination'} or r['numbering_subclass']=='numbering_content_rewrite': return 'reject'
    if r['semantic_subclass']=='near_synonym_or_shape_confusion' or r['numbering_subclass'] in {'enumeration_style_switch','arabic_digit_fix'} or r['boundary_subclass']=='boundary_plus_structure_symbol': return 'hard_review'
    if r['boundary_subclass'] in {'dual_boundary_edit','left_boundary_multi_char_fix','right_boundary_multi_char_fix'} or r['numbering_subclass']=='roman_numeral_fix' or r['is_length_changed']: return 'soft_review'
    if r['boundary_subclass'] in {'left_boundary_single_char_fix','right_boundary_single_char_fix'}: return 'accept'
    return 'accept'
def why(r):
    if r['proposed_backfill_action_v2']=='reject' and r['is_format_normalization']: return 'format-only rewrites are mostly harmful'
    if r['proposed_backfill_action_v2']=='reject' and r['semantic_subclass']!='none': return 'semantic rewrites show high harmful risk'
    if r['proposed_backfill_action_v2']=='reject': return 'number content rewrites may corrupt formulas and IDs'
    if r['proposed_backfill_action_v2']=='hard_review': return 'mixed-signal subclass with meaningful harmful mass'
    if r['proposed_backfill_action_v2']=='soft_review': return 'often helpful but structurally risky'
    return 'high-value local repair candidate'
def enrich(it,b):
    a=it.get('ocr_text','');v=it.get('final_text_if_upgraded') or it.get('vlm_raw_output') or it.get('final_text','');gt=it.get('gt','');ts=it.get('professional_terms',[]) or [];oed,ved=int(it.get('edit_distance_ocr',0)),int(it.get('edit_distance_final',0))
    r={'run_id':it.get('run_id',''),'analysis_split':it.get('split','test'),'analysis_scope':'exploratory_test_based','rule_freeze_allowed':False,'budget':b,'budget_tag':btag(b),'sample_id':it.get('sample_id',''),'domain':it.get('domain',''),'ocr_text':a,'vlm_raw_output':it.get('vlm_raw_output',''),'vlm_text':v,'gt':gt,'router_score':it.get('router_score',0.0),'edit_distance_ocr_to_gt':oed,'edit_distance_vlm_to_gt':ved,'edit_distance_delta':ved-oed,'is_length_changed':len(a)!=len(v),'is_numbering_changed':cset(a,NUMBERING)!=cset(v,NUMBERING),'is_symbol_changed':cset(a,STRUCT)!=cset(v,STRUCT),'is_term_changed':set(terms(ts,[a,gt]))!=set(terms(ts,[v,gt])),'is_boundary_edit':is_boundary(a,v),'is_format_normalization':is_fmt(a,v),'is_semantic_substitution':is_sem(a,v,gt),'edit_span_type':stype(a,v),'has_professional_terms':it.get('has_professional_terms',False),'professional_terms':'|'.join(ts),'backfill_status':it.get('backfill_status',''),'backfill_reason':it.get('backfill_reason','')}
    r['edit_gain_type']=gain(oed,ved);r['risk_tag']='|'.join([k for k,vv in [('boundary_edit',r['is_boundary_edit']),('numbering_change',r['is_numbering_changed']),('symbol_change',r['is_symbol_changed']),('term_change',r['is_term_changed']),('format_norm',r['is_format_normalization']),('semantic_sub',r['is_semantic_substitution']),('length_change',r['is_length_changed'])] if vv] or ['plain_local_edit'])
    r['edit_type']='numbering_rewrite' if r['is_numbering_changed'] else 'format_normalization' if r['is_format_normalization'] else 'term_edit' if r['is_term_changed'] else 'boundary_single_char_fix' if r['is_boundary_edit'] and r['edit_span_type']=='single_char' else 'boundary_edit' if r['is_boundary_edit'] else 'symbol_edit' if r['is_symbol_changed'] else 'semantic_substitution' if r['is_semantic_substitution'] else 'single_char_fix' if r['edit_span_type']=='single_char' else 'length_change' if r['is_length_changed'] else r['edit_span_type']
    r['semantic_subclass']=sem_sub(a,v,gt,ts);r['numbering_subclass']=num_sub(a,v);r['boundary_subclass']=bd_sub(a,v);r['proposed_backfill_action_v2']=action(r);r['rule_rationale']=why(r)
    return r
def export_pool(run_dir,b,out_dir):
    rows=[enrich(x,b) for x in read_jsonl(run_dir/f'M5_offline_budget_{btag(b)}_results.jsonl') if x.get('selected_for_upgrade')]
    if not rows:return []
    base=out_dir/f'm5_budget_{btag(b)}';fs=list(rows[0].keys())
    wcsv(base.with_name(base.name+'_edit_behavior_all.csv'),fs,rows)
    for k in ['beneficial','neutral','harmful']: wcsv(base.with_name(base.name+f'_edit_behavior_{k}.csv'),fs,[r for r in rows if r['edit_gain_type']==k])
    wcsv(base.with_name(base.name+'_case_pool_for_backfill.csv'),fs,rows)
    return rows
def type_stats(rows):
    acc=defaultdict(Counter)
    for r in rows: acc[(r['budget_tag'],r['edit_type'])][r['edit_gain_type']]+=1;acc[(r['budget_tag'],r['edit_type'])]['count']+=1
    out=[]
    for (bt,et),c in sorted(acc.items()):
        n=c['count'];out.append({'budget_tag':bt,'edit_type':et,'count':n,'beneficial_count':c['beneficial'],'neutral_count':c['neutral'],'harmful_count':c['harmful'],'beneficial_rate':round(c['beneficial']/n,6) if n else 0.0,'harmful_rate':round(c['harmful']/n,6) if n else 0.0})
    return out
def rule_table(rows):
    grp=defaultdict(list)
    for r in rows:
        for g,s in [('semantic',r['semantic_subclass']),('numbering',r['numbering_subclass']),('boundary',r['boundary_subclass'])]:
            if s!='none': grp[(g,s)].append(r)
    out=[]
    for (g,s),xs in sorted(grp.items()):
        n=len(xs);b=sum(r['edit_gain_type']=='beneficial' for r in xs);ne=sum(r['edit_gain_type']=='neutral' for r in xs);h=sum(r['edit_gain_type']=='harmful' for r in xs);a=Counter(r['proposed_backfill_action_v2'] for r in xs).most_common(1)[0][0]
        out.append({'feature_group':g,'feature_subclass':s,'count':n,'beneficial_count':b,'neutral_count':ne,'harmful_count':h,'beneficial_rate':round(b/n,6),'harmful_rate':round(h/n,6),'proposed_backfill_action_v2':a,'rule_rationale':xs[0]['rule_rationale'],'evidence_scope':'exploratory_test_based_only'})
    return out
def top_cases(rows,g,s,out_dir):
    key=f'{g}_subclass';xs=[r for r in rows if r.get(key)==s]
    if not xs:return
    fs=list(xs[0].keys())
    ben=sorted([r for r in xs if r['edit_gain_type']=='beneficial'],key=lambda r:(r['edit_distance_delta'],-float(r.get('router_score',0.0))))[:TOP_K_CASES]
    har=sorted([r for r in xs if r['edit_gain_type']=='harmful'],key=lambda r:(-r['edit_distance_delta'],-float(r.get('router_score',0.0))))[:TOP_K_CASES]
    wcsv(out_dir/f'{g}_{s}_top_beneficial.csv',fs,ben);wcsv(out_dir/f'{g}_{s}_top_harmful.csv',fs,har)

def main():
    ap=argparse.ArgumentParser(description='Phase B five-pool analysis + refined rule table');ap.add_argument('--phase_a_run_dir',required=True);ap.add_argument('--output_subdir',default='phase_b_five_pool');ap.add_argument('--budgets',default='0.05,0.10,0.20,0.60,1.00');a=ap.parse_args()
    run_dir=Path(a.phase_a_run_dir);out_dir=run_dir/a.output_subdir;out_dir.mkdir(parents=True,exist_ok=True);budgets=[float(x) for x in a.budgets.split(',') if x.strip()]
    rows=[]
    for b in budgets: rows+=export_pool(run_dir,b,out_dir)
    if not rows: raise ValueError('No upgraded samples found')
    fs=list(rows[0].keys());wcsv(out_dir/'m5_edit_behavior_all.csv',fs,rows)
    for k in ['beneficial','neutral','harmful']: wcsv(out_dir/f'm5_edit_behavior_{k}.csv',fs,[r for r in rows if r['edit_gain_type']==k])
    ts=type_stats(rows);rt=rule_table(rows)
    wcsv(out_dir/'m5_edit_type_stats.csv',list(ts[0].keys()) if ts else ['budget_tag'],ts)
    wcsv(out_dir/'backfill_v2_rule_table_refined.csv',list(rt[0].keys()) if rt else ['feature_group'],rt)
    wcsv(out_dir/'backfill_v2_action_mapping.csv',['feature_group','feature_subclass','proposed_backfill_action_v2','rule_rationale'],[{'feature_group':r['feature_group'],'feature_subclass':r['feature_subclass'],'proposed_backfill_action_v2':r['proposed_backfill_action_v2'],'rule_rationale':r['rule_rationale']} for r in rt])
    for g,subs in {'semantic':sorted({r['semantic_subclass'] for r in rows if r['semantic_subclass']!='none'}),'numbering':sorted({r['numbering_subclass'] for r in rows if r['numbering_subclass']!='none'}),'boundary':sorted({r['boundary_subclass'] for r in rows if r['boundary_subclass']!='none'})}.items():
        for s in subs: top_cases(rows,g,s,out_dir)
    jdump(out_dir/'phase_b_scope.json',{'analysis_scope':'exploratory_test_based','rule_freeze_allowed':False,'note':'Use for hypothesis generation only; freeze backfill_v2 on val or dedicated analysis split.','budgets':budgets,'phase':'B.5 refined rule table'})
    print(out_dir)

if __name__=='__main__': main()
