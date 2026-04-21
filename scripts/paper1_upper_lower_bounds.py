#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse,csv,json,time,sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import Levenshtein,yaml,requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.vlm_expert.gemini_expert import GeminiAgentB, GeminiConfig
from modules.vlm_expert import AgentBFactory


def norm(t):
    return '' if not t else t.translate(str.maketrans({'（':'(','）':')','【':'[','】':']','｛':'{','｝':'}','，':',','：':':','；':';','！':'!','？':'?','。':'.'}))

def rjsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]

def wjsonl(p, rows):
    with Path(p).open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def wcsv(p, rows):
    p = Path(p)
    if not rows:
        p.write_text('', encoding='utf-8')
        return
    fields, seen = [], set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k); fields.append(k)
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def cer_rows(rows, text_key):
    num = sum(Levenshtein.distance(norm(r.get(text_key,'')), norm(r.get('gt',''))) for r in rows)
    den = sum(len(norm(r.get('gt',''))) for r in rows)
    return round(num/den, 6) if den else 0.0

def acc_rows(rows, text_key):
    return round(sum(1 for r in rows if r.get(text_key,'') == r.get('gt','')) / len(rows), 6) if rows else 0.0

def resolve_image_path(workspace_root: Path, rel_path: str) -> str:
    rel = rel_path.replace('dataset/images/', 'data/l2w1data/images/')
    return str((workspace_root / rel).resolve())

def build_bonly_gemini(cfg):
    ab = cfg.get('agent_b', {})
    gc = GeminiConfig(
        base_url=ab.get('base_url', 'https://new.lemonapi.site/v1'),
        model_name=ab.get('model_name', 'gemini-3-flash-preview'),
        key_file=ab.get('key_file', 'key.txt'),
        provider_pool=ab.get('provider_pool', 'gemini_1x'),
        temperature=0.0,
        max_tokens=256,
        max_retries=ab.get('max_retries', 3),
        timeout=ab.get('timeout', 60),
    )
    agent = GeminiAgentB(gc)
    def call(image_path):
        prompt = (
            '你是中文单行手写/印刷文本识别助手。请直接读取图像中的单行文本并原样输出。\n'
            '要求：\n'
            '1. 只输出识别结果，不要解释。\n'
            '2. 尽量忠实保留符号、括号、箭头、编号、数字、上下标样式。\n'
            '3. 看不清时也不要编造上下文。\n'
            '4. 不要基于语言习惯润色改写。'
        )
        t0 = time.perf_counter()
        key = agent.config.key_manager.get_next_key()
        img64 = agent._encode_image(image_path)
        raw = agent._call_api(prompt, img64, key)
        txt = agent._parse_output(raw) if raw else ''
        return {'text': txt, 'latency_ms': round((time.perf_counter()-t0)*1000,3), 'error_type': 'none' if raw else 'empty_or_http'}
    return call, f"gemini:{gc.model_name}"

def build_bonly_local(cfg):
    cc = json.loads(json.dumps(cfg))
    ab = cc.setdefault('agent_b', {})
    ab['backend'] = 'local_vlm'
    expert = AgentBFactory.create(cc)
    def call(image_path):
        prompt = '请直接识别图像中的单行中文文本并原样输出。保留括号、箭头、编号、数字、符号；不要解释，不要润色。'
        t0 = time.perf_counter()
        try:
            txt = expert.chat_with_image(image_path, prompt).strip()
            txt = expert._parse_output(txt, '') if hasattr(expert, '_parse_output') else txt
            return {'text': txt, 'latency_ms': round((time.perf_counter()-t0)*1000,3), 'error_type': 'none'}
        except Exception as e:
            return {'text': '', 'latency_ms': round((time.perf_counter()-t0)*1000,3), 'error_type': type(e).__name__}
    info = expert.get_model_info()
    return call, f"local:{info.get('model_type')}:{Path(info.get('model_path','')).name}"

def main():
    p = argparse.ArgumentParser(description='paper1 upper/lower bounds runner')
    p.add_argument('--config', default='configs/router_config.yaml')
    p.add_argument('--maina_full_cache', default='paper1_runs/mainA/20260417_run112203/shared_repmodel_full_call_cache.jsonl')
    p.add_argument('--test_jsonl', default='data/l2w1data/test.jsonl')
    p.add_argument('--output_root', default='paper1_runs/upper_lower_bounds')
    p.add_argument('--workspace_root', default='G:/Code/PaddleOCR/L2W1')
    p.add_argument('--b_only_backend', choices=['gemini','local_vlm'], default='gemini')
    p.add_argument('--n_samples', type=int, default=None)
    p.add_argument('--local_model_type', default='qwen2.5_vl')
    p.add_argument('--local_model_path', default='./models/agent_b_vlm/qwen3-vl-8b-instruct')
    p.add_argument('--gemini_max_workers', type=int, default=300)
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    if a.b_only_backend == 'local_vlm':
        cfg = dict(cfg)
        cfg['agent_b'] = dict(cfg.get('agent_b', {}))
        cfg['agent_b']['model_type'] = a.local_model_type
        cfg['agent_b']['model_path'] = a.local_model_path

    full = rjsonl(a.maina_full_cache)
    tests = rjsonl(a.test_jsonl)
    gt_map = {r['sample_id']: r for r in tests}
    if a.n_samples:
        full = full[:a.n_samples]

    out = Path(a.output_root) / datetime.now().strftime('%Y%m%d_run%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    workspace_root = Path(a.workspace_root)

    a_only, a_plus_b = [], []
    for r in full:
        base = {'sample_id': r['sample_id'], 'domain': r.get('domain',''), 'image_path': r.get('image_path',''), 'gt': r['gt'], 'ocr_text': r['ocr_text']}
        a_only.append(dict(base, final_text=r['ocr_text'], system_name='A-Only', edit_distance_final=Levenshtein.distance(norm(r['ocr_text']), norm(r['gt']))))
        a_plus_b.append(dict(base, final_text=r['final_text_if_upgraded'], system_name='A+B Correction', edit_distance_final=Levenshtein.distance(norm(r['final_text_if_upgraded']), norm(r['gt']))))

    bonly_call, bonly_label = build_bonly_gemini(cfg) if a.b_only_backend == 'gemini' else build_bonly_local(cfg)

    b_only, error_stats = [], {}
    if a.b_only_backend == 'gemini':
        indexed = list(enumerate(full))
        results = [None] * len(full)
        with ThreadPoolExecutor(max_workers=a.gemini_max_workers) as ex:
            fut2idx = {}
            for idx, r in indexed:
                img = resolve_image_path(workspace_root, gt_map[r['sample_id']]['image_path'])
                fut = ex.submit(bonly_call, img)
                fut2idx[fut] = idx
            for fut in as_completed(fut2idx):
                idx = fut2idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    results[idx] = {'text': '', 'latency_ms': None, 'error_type': type(e).__name__}
        for r, rs in zip(full, results):
            txt = (rs or {}).get('text') or r['ocr_text']
            err = (rs or {}).get('error_type', 'none')
            error_stats[err] = error_stats.get(err, 0) + 1
            b_only.append({'sample_id': r['sample_id'], 'domain': r.get('domain',''), 'image_path': r.get('image_path',''), 'gt': r['gt'], 'ocr_text': r['ocr_text'], 'final_text': txt, 'system_name': 'B-Only Recognition', 'backend_label': bonly_label, 'latency_ms': (rs or {}).get('latency_ms'), 'error_type': err, 'edit_distance_final': Levenshtein.distance(norm(txt), norm(r['gt']))})
    else:
        for r in full:
            img = resolve_image_path(workspace_root, gt_map[r['sample_id']]['image_path'])
            rs = bonly_call(img)
            txt = rs.get('text') or r['ocr_text']
            err = rs.get('error_type', 'none')
            error_stats[err] = error_stats.get(err, 0) + 1
            b_only.append({'sample_id': r['sample_id'], 'domain': r.get('domain',''), 'image_path': r.get('image_path',''), 'gt': r['gt'], 'ocr_text': r['ocr_text'], 'final_text': txt, 'system_name': 'B-Only Recognition', 'backend_label': bonly_label, 'latency_ms': rs.get('latency_ms'), 'error_type': err, 'edit_distance_final': Levenshtein.distance(norm(txt), norm(r['gt']))})

    wjsonl(out / 'A_only_outputs.jsonl', a_only)
    wjsonl(out / 'A_plus_B_correction_outputs.jsonl', a_plus_b)
    wjsonl(out / 'B_only_recognition_outputs.jsonl', b_only)
    wcsv(out / 'b_only_error_stats.csv', [{'error_type': k, 'count': v} for k, v in sorted(error_stats.items())])
    wcsv(out / 'tab_upper_lower_bounds.csv', [
        {'system_name':'A-Only', 'CER': cer_rows(a_only,'final_text'), 'accuracy': acc_rows(a_only,'final_text'), 'n_samples': len(a_only), 'backend_label':'agent_a'},
        {'system_name':'B-Only Recognition', 'CER': cer_rows(b_only,'final_text'), 'accuracy': acc_rows(b_only,'final_text'), 'n_samples': len(b_only), 'backend_label': bonly_label},
        {'system_name':'A+B Correction', 'CER': cer_rows(a_plus_b,'final_text'), 'accuracy': acc_rows(a_plus_b,'final_text'), 'n_samples': len(a_plus_b), 'backend_label':'reuse_maina_gemini_cache'},
    ])

    b_map = {r['sample_id']: r for r in b_only}
    ab_map = {r['sample_id']: r for r in a_plus_b}
    case_pool = []
    for ao in a_only:
        sid = ao['sample_id']; bo = b_map[sid]; ab = ab_map[sid]
        case_pool.append({'sample_id': sid,'domain': ao['domain'],'ocr_text': ao['ocr_text'],'b_only_text': bo['final_text'],'a_plus_b_text': ab['final_text'],'gt': ao['gt'],'edit_distance_a_only': ao['edit_distance_final'],'edit_distance_b_only': bo['edit_distance_final'],'edit_distance_a_plus_b': ab['edit_distance_final'],'a_plus_b_better_than_b_only': ab['edit_distance_final'] < bo['edit_distance_final'],'a_plus_b_better_than_a_only': ab['edit_distance_final'] < ao['edit_distance_final'],'b_only_error_type': bo.get('error_type','none')})
    wcsv(out / 'upper_lower_case_pool.csv', case_pool)
    (out / 'manifest.json').write_text(json.dumps({'b_only_backend': a.b_only_backend, 'backend_label': bonly_label, 'n_samples': len(full), 'gemini_max_workers': a.gemini_max_workers, 'error_stats': error_stats}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out)

if __name__ == '__main__':
    main()
