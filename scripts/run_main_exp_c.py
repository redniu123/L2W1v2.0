#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, json, random, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.router.backfill import BackfillConfig, StrictBackfillController
from modules.router.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from modules.router.domain_knowledge import DomainKnowledgeEngine
from modules.router.uncertainty_router import BudgetControllerConfig
from modules.vlm_expert import AgentBFactory
from modules.vlm_expert.provider_pools import get_provider_pool
from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits
from scripts.run_efficiency_frontier import build_agent_b_callable, ensure_agent_a_result_schema, infer_all_samples
from scripts.run_online_budget_control import run_online_pipeline

MODELS = [
    {'exp_id': 'V1', 'name': 'Qwen3-VL-8B', 'backend': 'local_vlm', 'model_type': 'qwen2.5_vl', 'model_path': './models/agent_b_vlm/Qwen3-VL-8B'},
    {'exp_id': 'V2', 'name': 'InternVL3-8B', 'backend': 'local_vlm', 'model_type': 'internvl2_5', 'model_path': './models/agent_b_vlm/InternVL3-8B'},
    {'exp_id': 'V3', 'name': 'Gemini 3 Flash Preview', 'backend': 'gemini'},
    {'exp_id': 'V4', 'name': 'Claude Sonnet 4.6', 'backend': 'claude'},
]


def jdump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def jlines(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def wcsv(path, fieldnames, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})


def load_rows(args):
    cache = Path(args.cache_path)
    if args.use_cache and cache.exists() and not args.rebuild_cache:
        rows = ensure_agent_a_result_schema(json.loads(cache.read_text(encoding='utf-8')))
        return rows[:args.n_samples] if args.n_samples else rows
    samples = [json.loads(line) for line in Path(args.test_jsonl).read_text(encoding='utf-8').splitlines() if line.strip()]
    import argparse as _ap
    rec_args = _ap.Namespace(rec_model_dir=args.rec_model_dir, rec_char_dict_path=args.rec_char_dict_path, rec_image_shape='3, 48, 320', rec_batch_num=6, rec_algorithm='SVTR_LCNet', use_space_char=True, use_gpu=args.use_gpu, use_xpu=False, use_npu=False, use_mlu=False, use_metax_gpu=False, use_gcu=False, ir_optim=True, use_tensorrt=False, min_subgraph_size=15, precision='fp32', gpu_mem=500, gpu_id=0, enable_mkldnn=None, cpu_threads=10, warmup=False, benchmark=False, save_log_path='./log_output/', show_log=False, use_onnx=False, max_batch_size=10, return_word_box=False, drop_score=0.5, max_text_length=25, rec_image_inverse=True, use_det=False, det_model_dir='')
    recognizer = TextRecognizerWithLogits(rec_args)
    domain_engine = DomainKnowledgeEngine({'geology': args.geo_dict, 'finance': args.finance_dict, 'medicine': args.medicine_dict})
    rows = ensure_agent_a_result_schema(infer_all_samples(samples, recognizer, domain_engine, None, args.image_root))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    return rows[:args.n_samples] if args.n_samples else rows


def build_claude_callable(config):
    pool = get_provider_pool('claude_sonnet_46', config['agent_b'].get('key_file', 'key.txt'))
    base_url = pool.base_url
    api_key = pool.keys[0]
    model_name = pool.model_name
    timeout = config['agent_b'].get('timeout', 360)

    def call(prompt):
        t0 = time.perf_counter()
        ocr_text = prompt.get('T_A', '')
        payload = {'model': model_name, 'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': f'你是中文OCR纠错助手。仅输出修正后的完整文本。原始OCR文本：{ocr_text}'}]}], 'temperature': 0.1, 'max_tokens': 256}
        try:
            resp = requests.post(f'{base_url}/chat/completions', headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}, json=payload, timeout=timeout)
            resp.raise_for_status()
            text = resp.json()['choices'][0]['message']['content'].strip().split('\n')[0].strip()
            return {'corrected_text': text or ocr_text, 'latency_ms': round((time.perf_counter() - t0) * 1000, 3), 'token_usage': None, 'error_type': 'none'}
        except Exception as e:
            return {'corrected_text': ocr_text, 'latency_ms': round((time.perf_counter() - t0) * 1000, 3), 'token_usage': None, 'error_type': type(e).__name__}

    call._backend = 'claude'
    call._model_label = f'claude:{model_name}'
    call._supports_parallel = False
    call._max_concurrency = 1
    return call


def build_model_callable(spec, config):
    if spec['backend'] == 'gemini':
        return build_agent_b_callable(config)
    if spec['backend'] == 'claude':
        return build_claude_callable(config)
    cfg = json.loads(json.dumps(config))
    cfg['agent_b'] = {'skip': False, 'backend': 'local_vlm', 'model_type': spec['model_type'], 'model_path': spec['model_path'], 'torch_dtype': 'float16', 'max_new_tokens': 128}
    expert = AgentBFactory.create(cfg)

    def local_call(prompt):
        t0 = time.perf_counter()
        manifest = {'ocr_text': prompt.get('T_A', ''), 'suspicious_index': prompt.get('min_conf_idx', -1) or -1, 'suspicious_char': '', 'risk_level': 'medium'}
        try:
            result = expert.process_hard_sample(prompt.get('image_path') or prompt.get('img_path', ''), manifest)
            corrected = result.get('corrected_text', prompt.get('T_A', '')) if isinstance(result, dict) else result
            token_usage = result.get('token_usage') if isinstance(result, dict) else None
            error_type = result.get('error_type', 'none') if isinstance(result, dict) else 'none'
            return {'corrected_text': corrected, 'latency_ms': round((time.perf_counter() - t0) * 1000, 3), 'token_usage': token_usage, 'error_type': error_type}
        except Exception as e:
            return {'corrected_text': prompt.get('T_A', ''), 'latency_ms': round((time.perf_counter() - t0) * 1000, 3), 'token_usage': None, 'error_type': type(e).__name__}

    local_call._backend = 'local_vlm'
    local_call._model_label = f"local_vlm:{spec['name']}"
    local_call._supports_parallel = False
    local_call._max_concurrency = 1
    return local_call


def prompt_stub():
    class PromptStub:
        def generate_targeted_correction_prompt(self, **kw):
            return {'T_A': kw['T_A'], 'min_conf_idx': kw.get('min_conf_idx'), 'image_path': kw.get('image_path'), 'domain': kw.get('domain')}
    return PromptStub()


def patch_rows(rows, spec, budget, run_id, prompt_version):
    for row in rows:
        row.update({'exp_id': spec['exp_id'], 'system_name': 'SH-DA++', 'model_name': spec['name'], 'budget': budget, 'run_id': run_id, 'prompt_version': prompt_version, 'router_name': 'GCR'})
    return rows


def main():
    p = argparse.ArgumentParser(description='Run Main Experiment C')
    p.add_argument('--config', default='configs/router_config.yaml')
    p.add_argument('--test_jsonl', default='data/l2w1data/test.jsonl')
    p.add_argument('--image_root', default='data/l2w1data/images')
    p.add_argument('--rec_model_dir', default='./models/agent_a_ppocr/PP-OCRv5_server_rec_infer')
    p.add_argument('--rec_char_dict_path', default='ppocr/utils/ppocrv5_dict.txt')
    p.add_argument('--geo_dict', default='data/dicts/Geology.txt')
    p.add_argument('--finance_dict', default='data/dicts/Finance.txt')
    p.add_argument('--medicine_dict', default='data/dicts/Medicine.txt')
    p.add_argument('--output_dir', default='results/expriments/exC/02_runs')
    p.add_argument('--cache_path', default='results/expriments/exC/02_runs/agent_a_cache.json')
    p.add_argument('--budgets', nargs='+', type=float, default=[0.10, 0.20, 0.30])
    p.add_argument('--models', nargs='+', default=['V1', 'V2', 'V3', 'V4'])
    p.add_argument('--n_samples', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--use_gpu', action='store_true', default=False)
    p.add_argument('--use_cache', action='store_true', default=True)
    p.add_argument('--rebuild_cache', action='store_true', default=False)
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    prompt_version = config.get('prompt_version') or config.get('mainline', {}).get('prompt_version', 'prompt_v1.1')
    rows = load_rows(args)
    backfill = StrictBackfillController(BackfillConfig())
    run_id = datetime.now().strftime('%Y%m%d_run%H%M%S')
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    active_models = [m for m in MODELS if m['exp_id'] in set(args.models)]
    (run_dir / 'config_snapshot.yaml').write_text(yaml.safe_dump({'run_id': run_id, 'budgets': args.budgets, 'models': active_models, 'prompt_version': prompt_version, 'router_name': 'GCR', 'system_name': 'SH-DA++'}, allow_unicode=True, sort_keys=False), encoding='utf-8')

    summary_rows, metric_rows, failure_rows = [], [], []
    for spec in active_models:
        agent_b = build_model_callable(spec, config)
        for budget in args.budgets:
            cb_cfg = (config or {}).get('sh_da_v4', {}).get('circuit_breaker', {})
            cb = CircuitBreaker(CircuitBreakerConfig(enabled=cb_cfg.get('enabled', True), min_samples=cb_cfg.get('min_samples', 20), rejection_rate_threshold=cb_cfg.get('rejection_rate_threshold', 0.60), cooldown_steps=cb_cfg.get('cooldown_steps', 50)))
            result = run_online_pipeline('GCR', budget, rows, None, backfill, prompt_stub(), agent_b, BudgetControllerConfig(target_budget=budget), cb, run_id=run_id, prompt_version=prompt_version, agent_b_label=spec['name'], skip_backfill=False, domain_prompt=True)
            patched = patch_rows(result['per_sample'], spec, budget, run_id, prompt_version)
            tag = spec['name'].replace(' ', '').replace('-', '').replace('.', '')
            jlines(run_dir / f"{spec['exp_id']}_{tag}_online_budget_{int(round(budget * 100))}_results.jsonl", patched)
            row = {'run_id': run_id, 'exp_group': 'mainC', 'exp_id': spec['exp_id'], 'system_name': 'SH-DA++', 'best_router_name': 'GCR', 'model_name': spec['name'], 'budget': budget, 'target_call_rate': budget, 'actual_call_rate': result['summary']['Actual_Call_Rate'], 'call_rate_valid': abs(float(result['summary']['Actual_Call_Rate']) - budget) <= 0.005, 'CER': result['summary']['Overall_CER'], 'BoundaryDeletionRecallAtB': result['summary']['Boundary_Deletion_Recall@B'], 'SubstitutionCER': result['summary']['Substitution_CER'], 'CVR': result['summary']['CVR'], 'AER': result['summary']['AER'], 'p95_latency_ms': result['summary']['P95_Latency_MS'], 'avg_token_usage': result['summary']['Avg_Token_Usage'], 'num_samples': result['summary']['N_valid']}
            summary_rows.append(row)
            metric_rows.append({'exp_id': spec['exp_id'], 'model_name': spec['name'], 'budget': budget, 'metrics': result['summary']})
            for item in patched:
                if item.get('is_correct_ocr') is False and item.get('is_correct_final') is False:
                    failure_rows.append({'exp_id': spec['exp_id'], 'model_name': spec['name'], 'budget': budget, 'sample_id': item.get('sample_id', ''), 'domain': item.get('domain', ''), 'ocr_text': item.get('ocr_text', ''), 'final_text': item.get('final_text', ''), 'gt': item.get('gt', ''), 'backfill_status': item.get('backfill_status', ''), 'backfill_reason': item.get('backfill_reason', '')})

    wcsv(run_dir / 'summary.csv', list(summary_rows[0].keys()) if summary_rows else ['run_id'], summary_rows)
    wcsv(run_dir / 'failure_cases.csv', list(failure_rows[0].keys()) if failure_rows else ['exp_id'], failure_rows)
    jdump(run_dir / 'metrics_summary.json', {'run_id': run_id, 'exp_group': 'mainC', 'results': metric_rows})
    print(run_dir)


if __name__ == '__main__':
    main()
