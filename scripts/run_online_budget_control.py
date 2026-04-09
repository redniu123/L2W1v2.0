#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""正式在线预算实验：逐样本使用 OnlineBudgetController.step(q)。"""

import argparse, csv, json, random, sys
from datetime import datetime
from pathlib import Path

import numpy as np, yaml, Levenshtein
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.router.uncertainty_router import BudgetControllerConfig, OnlineBudgetController
from modules.router.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from scripts.run_efficiency_frontier import build_agent_b_callable, infer_all_samples, summarize_extended_metrics


def build_scores(strategy, all_results, router):
    scores = []
    for r in all_results:
        if strategy == 'GCR':
            scores.append(1.0 - r['conf'])
        elif strategy == 'BAUR':
            d = router.route(r['boundary_stats'], r['top2_info'], r_d=0.0, agent_a_text=r['T_A'])
            scores.append(max(d.s_b, d.s_a))
        elif strategy == 'DAR':
            d = router.route(r['boundary_stats'], r['top2_info'], r_d=r['r_d'], agent_a_text=r['T_A'])
            scores.append(d.q)
        else:
            d = router.route(r['boundary_stats'], r['top2_info'], r_d=0.0, agent_a_text=r['T_A'])
            scores.append(max(d.s_b, d.s_a))
    return scores


def run_online_pipeline(strategy, target_budget, all_results, router, backfill_controller, prompter, agent_b_callable, budget_cfg, circuit_breaker, run_id='', prompt_version='prompt_v1.0', agent_b_label='configured_agent_b'):
    from modules.router.backfill import RouteType
    ctrl = OnlineBudgetController(budget_cfg)
    scores = build_scores(strategy, all_results, router)
    cer_num = gt_len = n_upgraded = n_accepted = n_rejected = 0
    per_sample, backfill_log = [], []
    for i, r in enumerate(tqdm(all_results, desc=f'{strategy} online B={target_budget:.2f}', leave=False)):
        T_A, T_GT, q = r['T_A'], r['T_GT'], float(scores[i])
        upgrade, bd = ctrl.step(q)
        if upgrade and not circuit_breaker.allow_upgrade():
            upgrade = False
            bd = {**bd, 'circuit_breaker_blocked': True}
        else:
            bd = {**bd, 'circuit_breaker_blocked': False}
        lam = bd.get('lambda_before', ctrl.current_lambda)
        win = bd.get('actual_budget', ctrl.actual_budget)
        vlm_raw_output = final_text_if_upgraded = ''
        latency_ms = None
        token_usage = None
        error_type = 'not_upgraded'
        backfill_status = backfill_reason = 'not_upgraded'
        cb_stats = circuit_breaker.step_without_call()
        if upgrade:
            n_upgraded += 1
            domain_label = {'geology': '地质勘探', 'finance': '金融财会', 'medicine': '医学'}.get(r.get('domain', 'geology')) if strategy == 'SH-DA++' else None
            prompt = prompter.generate_targeted_correction_prompt(T_A=T_A, min_conf_idx=r['min_conf_idx'], domain=domain_label, image_path=r['img_path'])
            prompt['T_A'], prompt['min_conf_idx'], prompt['image_path'] = T_A, r['min_conf_idx'], r['img_path']
            agent_b_result = agent_b_callable(prompt)
            T_cand = agent_b_result.get('corrected_text', T_A)
            vlm_raw_output = T_cand
            latency_ms = agent_b_result.get('latency_ms')
            token_usage = agent_b_result.get('token_usage')
            error_type = agent_b_result.get('error_type', 'none')
            if strategy == 'BAUR-only':
                T_final = T_cand if isinstance(T_cand, str) and T_cand else T_A
                final_text_if_upgraded, backfill_status, backfill_reason = T_final, 'skipped', 'baur_only_no_backfill'
                cb_stats = circuit_breaker.observe(rejected=False)
                if T_final != T_A: n_accepted += 1
            else:
                bf = backfill_controller.apply_backfill(T_A=T_A, T_cand=T_cand, route_type=RouteType.BOUNDARY)
                T_final = bf.T_final
                final_text_if_upgraded = T_final
                if bf.is_rejected:
                    n_rejected += 1; backfill_status, backfill_reason = 'rejected', bf.rejection_reason.value; cb_stats = circuit_breaker.observe(rejected=True)
                else:
                    backfill_status, backfill_reason = 'accepted', bf.rejection_reason.value; cb_stats = circuit_breaker.observe(rejected=False)
                    if T_final != T_A: n_accepted += 1
        else:
            T_final = T_A
        cer_num += Levenshtein.distance(T_final, T_GT); gt_len += len(T_GT)
        row = {'sample_id': r.get('sample_id',''), 'image_path': r.get('image_path',''), 'source_image_id': r.get('source_image_id',''), 'domain': r.get('domain','geology'), 'split': r.get('split','test'), 'gt': T_GT, 'ocr_text': T_A, 'router_name': strategy, 'router_score': round(q,6), 'budget': target_budget, 'budget_mode': 'online_control', 'selected_for_upgrade': upgrade, 'lambda_current': round(lam,6), 'actual_budget_window': round(win,6), 'circuit_breaker_open': cb_stats.get('is_open', False), 'circuit_breaker_blocked': bd.get('circuit_breaker_blocked', False), 'vlm_model': agent_b_label, 'prompt_version': prompt_version, 'vlm_raw_output': vlm_raw_output, 'latency_ms': latency_ms, 'token_usage': token_usage, 'error_type': error_type, 'final_text_if_upgraded': final_text_if_upgraded, 'final_text': T_final, 'backfill_status': backfill_status, 'backfill_reason': backfill_reason, 'is_correct_ocr': T_A == T_GT, 'is_correct_final': T_final == T_GT, 'edit_distance_ocr': Levenshtein.distance(T_A, T_GT), 'edit_distance_final': Levenshtein.distance(T_final, T_GT), 'run_id': run_id}
        per_sample.append(row)
        if upgrade:
            backfill_log.append({'sample_id': row['sample_id'], 'router_name': strategy, 'budget': target_budget, 'lambda_current': row['lambda_current'], 'actual_budget_window': row['actual_budget_window'], 'ocr_text': T_A, 'vlm_raw_output': vlm_raw_output, 'latency_ms': latency_ms, 'token_usage': token_usage, 'error_type': error_type, 'final_text': T_final, 'backfill_status': backfill_status, 'backfill_reason': backfill_reason, 'run_id': run_id})
    extended_metrics = summarize_extended_metrics(per_sample)
    return {'summary': {'Strategy': strategy, 'Target_Budget': target_budget, 'Actual_Call_Rate': round(n_upgraded / len(all_results), 4) if all_results else 0.0, 'Overall_CER': round(cer_num / gt_len, 6) if gt_len else 0.0, 'Boundary_Deletion_Recall@B': extended_metrics['Boundary_Deletion_Recall@B'], 'Substitution_CER': extended_metrics['Substitution_CER'], 'AER': round(n_accepted / n_upgraded, 4) if n_upgraded else 0.0, 'CVR': round(n_rejected / n_upgraded, 4) if n_upgraded else 0.0, 'N_valid': len(all_results)}, 'budget_stats': ctrl.get_stats(), 'per_sample': per_sample, 'backfill_log': backfill_log}


def main():
    p = argparse.ArgumentParser(description='SH-DA++ Online Budget Control')
    p.add_argument('--config', default='configs/router_config.yaml'); p.add_argument('--test_jsonl', default='data/l2w1data/test.jsonl')
    p.add_argument('--image_root', default='data/l2w1data/images'); p.add_argument('--rec_model_dir', default='./models/agent_a_ppocr/PP-OCRv5_server_rec_infer')
    p.add_argument('--rec_char_dict_path', default='ppocr/utils/ppocrv5_dict.txt'); p.add_argument('--geo_dict', default='data/dicts/Geology.txt')
    p.add_argument('--finance_dict', default='data/dicts/Finance.txt'); p.add_argument('--medicine_dict', default='data/dicts/Medicine.txt')
    p.add_argument('--output_dir', default='results/stage2_v51_online'); p.add_argument('--strategy', default='SH-DA++', choices=['GCR','BAUR','DAR','BAUR-only','SH-DA++'])
    p.add_argument('--target_budget', type=float, default=0.10); p.add_argument('--seed', type=int, default=42); p.add_argument('--n_samples', type=int, default=None)
    p.add_argument('--use_gpu', action='store_true', default=False); p.add_argument('--use_cache', action='store_true', default=False); p.add_argument('--rebuild_cache', action='store_true', default=False)
    p.add_argument('--prompt_version', default=None, help='Override frozen prompt version')
    args = p.parse_args(); random.seed(args.seed); np.random.seed(args.seed)
    with open(args.config, 'r', encoding='utf-8') as f: config = yaml.safe_load(f)
    prompt_version = args.prompt_version or config.get('prompt_version', 'prompt_v1.0')
    mainline_agent_b = config.get('mainline_agent_b', 'configured_agent_b')
    import argparse as _ap
    rec_args = _ap.Namespace(rec_model_dir=args.rec_model_dir, rec_char_dict_path=args.rec_char_dict_path, rec_image_shape='3, 48, 320', rec_batch_num=6, rec_algorithm='SVTR_LCNet', use_space_char=True, use_gpu=args.use_gpu, use_xpu=False, use_npu=False, use_mlu=False, use_metax_gpu=False, use_gcu=False, ir_optim=True, use_tensorrt=False, min_subgraph_size=15, precision='fp32', gpu_mem=500, gpu_id=0, enable_mkldnn=None, cpu_threads=10, warmup=False, benchmark=False, save_log_path='./log_output/', show_log=False, use_onnx=False, max_batch_size=10, return_word_box=False, drop_score=0.5, max_text_length=25, rec_image_inverse=True, use_det=False, det_model_dir='')
    from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits
    from modules.router.sh_da_router import SHDARouter
    from modules.router.backfill import BackfillConfig, StrictBackfillController
    from modules.vlm_expert.constrained_prompter import ConstrainedPrompter
    from modules.router.domain_knowledge import DomainKnowledgeEngine
    recognizer = TextRecognizerWithLogits(rec_args); router = SHDARouter.from_yaml(args.config); backfill_controller = StrictBackfillController(BackfillConfig()); prompter = ConstrainedPrompter(); domain_engine = DomainKnowledgeEngine({'geology': args.geo_dict, 'finance': args.finance_dict, 'medicine': args.medicine_dict}); agent_b_callable = build_agent_b_callable(config)
    samples = [json.loads(line) for line in Path(args.test_jsonl).read_text(encoding='utf-8').splitlines() if line.strip()]
    cache_path = Path(args.output_dir) / 'agent_a_cache.json'
    if args.use_cache and not args.rebuild_cache and cache_path.exists(): all_results = json.loads(cache_path.read_text(encoding='utf-8'))
    else:
        all_results = infer_all_samples(samples, recognizer, domain_engine, None, args.image_root); Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        cache_data = []
        for r in all_results:
            rc = dict(r)
            for k in ('top2_info','boundary_stats'):
                if isinstance(rc.get(k), dict): rc[k] = {kk: (vv.tolist() if hasattr(vv,'tolist') else vv) for kk, vv in rc[k].items()}
            cache_data.append(rc)
        cache_path.write_text(json.dumps(cache_data, ensure_ascii=False), encoding='utf-8')
    if args.n_samples and args.n_samples < len(all_results): all_results = all_results[:args.n_samples]
    run_id = datetime.now().strftime('%Y%m%d_run%H%M%S'); run_dir = Path(args.output_dir) / run_id; run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config_snapshot.yaml').write_text(yaml.safe_dump({'run_id': run_id, 'args': vars(args), 'config': config}, allow_unicode=True, sort_keys=False), encoding='utf-8')
    bc = (config or {}).get('sh_da_v4', {}).get('budget_controller', {})
    budget_cfg = BudgetControllerConfig(window_size=bc.get('window_size', 500), k=bc.get('k', 0.01), lambda_min=bc.get('lambda_min', 0.0), lambda_max=bc.get('lambda_max', 2.0), lambda_init=bc.get('lambda_init', 0.5), target_budget=args.target_budget)
    cb_cfg = (config or {}).get('sh_da_v4', {}).get('circuit_breaker', {})
    circuit_breaker = CircuitBreaker(CircuitBreakerConfig(enabled=cb_cfg.get('enabled', True), min_samples=cb_cfg.get('min_samples', 20), rejection_rate_threshold=cb_cfg.get('rejection_rate_threshold', 0.60), cooldown_steps=cb_cfg.get('cooldown_steps', 50)))
    result = run_online_pipeline(args.strategy, args.target_budget, all_results, router, backfill_controller, prompter, agent_b_callable, budget_cfg, circuit_breaker, run_id=run_id, prompt_version=prompt_version, agent_b_label=mainline_agent_b)
    with open(run_dir / 'summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['Strategy','Target_Budget','Actual_Call_Rate','Overall_CER','Boundary_Deletion_Recall@B','Substitution_CER','AER','CVR','N_valid']); w.writeheader(); w.writerow(result['summary'])
    (run_dir / 'metrics_summary.json').write_text(json.dumps([result['summary']], ensure_ascii=False, indent=2), encoding='utf-8')
    (run_dir / 'budget_stability.json').write_text(json.dumps(result['budget_stats'], ensure_ascii=False, indent=2), encoding='utf-8')
    (run_dir / 'circuit_breaker.json').write_text(json.dumps(circuit_breaker.get_stats(), ensure_ascii=False, indent=2), encoding='utf-8')
    jp = run_dir / f"online_budget_{int(round(args.target_budget*100)):02d}_{args.strategy}.jsonl"
    with open(jp, 'w', encoding='utf-8') as f:
        for item in result['per_sample']: f.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(run_dir / 'backfill_log.jsonl', 'w', encoding='utf-8') as f:
        for item in result['backfill_log']: f.write(json.dumps(item, ensure_ascii=False) + '\n')
    failures, domain_stats = [], {}
    for item in result['per_sample']:
        key = (item.get('router_name','unknown'), item.get('domain','unknown'), str(item.get('budget',0.0))); stat = domain_stats.setdefault(key, {'total':0,'correct_final':0}); stat['total'] += 1; stat['correct_final'] += 1 if item.get('is_correct_final') else 0
        if item.get('is_correct_ocr') is False and item.get('is_correct_final') is False: failures.append({'sample_id': item.get('sample_id',''), 'router_name': item.get('router_name',''), 'budget': item.get('budget',0.0), 'domain': item.get('domain',''), 'ocr_text': item.get('ocr_text',''), 'final_text': item.get('final_text',''), 'gt': item.get('gt',''), 'backfill_status': item.get('backfill_status',''), 'backfill_reason': item.get('backfill_reason',''), 'edit_distance_final': item.get('edit_distance_final',0)})
    with open(run_dir / 'failure_cases.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['sample_id','router_name','budget','domain','ocr_text','final_text','gt','backfill_status','backfill_reason','edit_distance_final']); w.writeheader(); [w.writerow(r) for r in failures]
    with open(run_dir / 'domain_breakdown.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['router_name','domain','budget','n_samples','final_accuracy']); w.writeheader(); [w.writerow({'router_name': rn, 'domain': d, 'budget': b, 'n_samples': s['total'], 'final_accuracy': round((s['correct_final']/s['total']) if s['total'] else 0.0, 6)}) for (rn, d, b), s in sorted(domain_stats.items())]
    print(f'Done: {run_dir}')


if __name__ == '__main__':
    main()
