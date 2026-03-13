#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1 Phase 2 Task 2: Agent B 单样本冒烟测试

从 Val 集中手动挑选 3 个样本，分别触发 BOUNDARY / AMBIGUITY / BOTH 路径，
全程追踪打印：route_type、idx_susp、Prompt、T_cand、T_final 及拒改原因。

Agent B 调用模式：
  - 若 router_config.yaml 中 agent_b.skip=false，且本地有 Qwen2.5-VL，使用真实 Agent B
  - 否则使用 Mock 模式（直接返回 T_A 不改），确保流程可测试
"""

import json
import sys
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.router.backfill import BackfillConfig, RouteType, StrictBackfillController
from modules.router.sh_da_router import SHDARouter
from modules.vlm_expert.constrained_prompter import ConstrainedPrompter


SEPARATOR = "=" * 60


def mock_agent_b(prompt: dict, image_path: str) -> str:
    """
    Mock Agent B：直接返回 T_A（不改动）
    用于在没有 Qwen2.5-VL 的情况下测试完整流程
    """
    return prompt["user_prompt"].split("\n")[1].strip()


def load_agent_b(config: dict):
    """
    根据配置加载 Agent B 或返回 Mock

    Returns:
        callable: agent_b(prompt_dict, image_path) -> str
    """
    agent_b_cfg = config.get("agent_b", {})
    skip = agent_b_cfg.get("skip", True)

    if skip:
        print("[Agent B] skip=true，使用 Mock 模式（直接返回 T_A）")
        return mock_agent_b

    model_path = agent_b_cfg.get("model_path", "")
    if not model_path or not Path(model_path).exists():
        print(f"[Agent B] 模型路径不存在: {model_path}，降级为 Mock 模式")
        return mock_agent_b

    # 真实 Agent B（Qwen2.5-VL）
    try:
        from modules.vlm_expert.agent_b import AgentB
        agent = AgentB(model_path=model_path)
        print(f"[Agent B] 真实模型已加载: {model_path}")
        return agent.generate
    except Exception as e:
        print(f"[Agent B] 加载失败: {e}，降级为 Mock 模式")
        return mock_agent_b


def run_smoke_test_sample(
    sample_idx: int,
    image_path: str,
    T_GT: str,
    target_route: str,
    recognizer,
    router: SHDARouter,
    prompter: ConstrainedPrompter,
    backfill: StrictBackfillController,
    agent_b_callable,
    image_root: str,
) -> None:
    """
    对单个样本执行完整的流转测试
    """
    import cv2

    print(f"\n{SEPARATOR}")
    print(f"[冒烟测试 {sample_idx}] 目标路径: {target_route}")
    print(f"  图像: {image_path}")
    print(f"  GT:   {T_GT}")
    print(SEPARATOR)

    # 解析图像路径
    img_path = Path(image_path)
    if not img_path.is_absolute():
        img_path = Path(image_root).resolve() / img_path

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [错误] 无法读取图像: {img_path}")
        return

    # Agent A 推理
    output = recognizer([img])
    if not output or not output.get("results"):
        print("  [错误] Agent A 推理失败")
        return

    T_A, conf = output["results"][0]
    boundary_stats_list = output.get("boundary_stats", [])
    top2_info_list = output.get("top2_info", [])
    boundary_stats = boundary_stats_list[0] if boundary_stats_list else None
    top2_info = top2_info_list[0] if top2_info_list else None

    print(f"\n--- Step 1: Agent A 输出 ---")
    print(f"  T_A:  {T_A}")
    print(f"  conf: {conf:.4f}")

    # Router 决策
    from modules.router.domain_knowledge import DomainKnowledgeEngine
    # r_d 在此处已嵌入 router 评分中（简化：直接传 0）
    decision = router.route(
        boundary_stats=boundary_stats or {},
        top2_info=top2_info or {},
        r_d=0.0,
        agent_a_text=T_A,
    )

    print(f"\n--- Step 2: Router 决策 ---")
    print(f"  upgrade:      {decision.upgrade}")
    print(f"  route_type:   {decision.route_type.value}")
    print(f"  s_b:          {decision.s_b:.4f}")
    print(f"  s_a:          {decision.s_a:.4f}")
    print(f"  q:            {decision.q:.4f}")
    print(f"  lambda:       {decision.lambda_current:.4f}")
    print(f"  idx_susp:     {decision.idx_susp}")
    print(f"  top2_chars:   {decision.top2_chars}")

    if not decision.upgrade:
        print("\n  [信息] Router 判定无需升级，直接输出 T_A")
        print(f"  T_final: {T_A}")
        return

    # 生成 Prompt
    route_type = decision.route_type
    if route_type == RouteType.BOUNDARY:
        prompt = prompter.generate_boundary_prompt(T_A, str(img_path))
    elif route_type == RouteType.AMBIGUITY and decision.idx_susp is not None:
        prompt = prompter.generate_ambiguity_prompt(
            T_A, decision.idx_susp, decision.top2_chars or [T_A[decision.idx_susp], "？"],
            str(img_path)
        )
    else:
        # BOTH 或 fallback
        prompt = prompter.generate_boundary_prompt(T_A, str(img_path))

    print(f"\n--- Step 3: Prompt ---")
    print(f"  prompt_type:  {prompt.get('prompt_type', 'unknown')}")
    print(f"  system_prompt (前80字): {prompt['system_prompt'][:80]}...")
    print(f"  user_prompt:\n{prompt['user_prompt']}")

    # Agent B 推理
    T_cand = agent_b_callable(prompt, str(img_path))
    print(f"\n--- Step 4: Agent B 返回 ---")
    print(f"  T_cand: {T_cand}")

    # Backfill
    backfill_result = backfill.apply(
        T_A=T_A,
        T_cand=T_cand,
        route_type=route_type,
        idx_susp=decision.idx_susp,
        top2_chars=decision.top2_chars,
    )

    print(f"\n--- Step 5: Backfill 结果 ---")
    print(f"  T_final:          {backfill_result.T_final}")
    print(f"  is_accepted:      {backfill_result.is_accepted}")
    print(f"  rejection_reason: {backfill_result.rejection_reason}")
    print(f"  edit_distance:    {backfill_result.edit_distance}")
    print(f"\n  GT 对比:          {T_GT}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SH-DA++ v5.1: Agent B 冒烟测试")
    parser.add_argument("--config", type=str, default="configs/router_config.yaml")
    parser.add_argument("--val_jsonl", type=str, default="data/raw/hctr_riskbench/val.jsonl")
    parser.add_argument("--metadata_jsonl", type=str,
                        default="results/stage2_v51/metadata_val.jsonl")
    parser.add_argument("--image_root", type=str, default="data/geo")
    parser.add_argument("--rec_model_dir", type=str,
                        default="./models/agent_a_ppocr/PP-OCRv5_server_rec_infer")
    parser.add_argument("--rec_char_dict_path", type=str,
                        default="ppocr/utils/ppocrv5_dict.txt")
    args = parser.parse_args()

    # 读取配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(SEPARATOR)
    print("SH-DA++ v5.1: Agent B 单样本冒烟测试")
    print(SEPARATOR)

    # 初始化组件
    from tools.infer.utility import init_args
    import argparse as ap
    rec_args = ap.Namespace(
        rec_model_dir=args.rec_model_dir,
        rec_char_dict_path=args.rec_char_dict_path,
        rec_image_shape="3, 48, 320",
        rec_batch_num=6,
        rec_algorithm="SVTR_LCNet",
        use_space_char=True,
        use_gpu=True,
        use_xpu=False,
        use_npu=False,
        use_mlu=False,
        use_metax_gpu=False,
        use_gcu=False,
        ir_optim=True,
        use_tensorrt=False,
        min_subgraph_size=15,
        precision="fp32",
        gpu_mem=500,
        gpu_id=0,
        enable_mkldnn=None,
        cpu_threads=10,
        warmup=False,
        benchmark=False,
        save_log_path="./log_output/",
        show_log=True,
        use_onnx=False,
        max_batch_size=10,
        return_word_box=False,
        drop_score=0.5,
        max_text_length=25,
        rec_image_inverse=True,
        use_det=False,
        det_model_dir="",
    )

    from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits
    print("初始化 Agent A...")
    recognizer = TextRecognizerWithLogits(rec_args)

    print("初始化 Router...")
    router = SHDARouter.from_yaml(args.config)

    prompter = ConstrainedPrompter()
    backfill = StrictBackfillController(BackfillConfig())
    agent_b = load_agent_b(config)

    # 读取 metadata，找出不同路径的样本
    meta_samples = []
    with open(args.metadata_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                meta_samples.append(json.loads(line))

    # 读取 val.jsonl，建立 image -> sample 映射
    val_map = {}
    with open(args.val_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                s = json.loads(line)
                key = s.get("image") or s.get("image_path", "")
                val_map[key] = s

    # 策略：挑选前 3 个有效样本进行测试（不区分路径，由 Router 自然分配）
    # 实际路径由 Router 实时决策，不人工指定
    test_samples = [m for m in meta_samples if m.get("T_GT")][:3]

    if len(test_samples) < 3:
        print(f"[警告] 只找到 {len(test_samples)} 个有效样本")

    for i, meta in enumerate(test_samples):
        run_smoke_test_sample(
            sample_idx=i + 1,
            image_path=meta["image"],
            T_GT=meta["T_GT"],
            target_route="AUTO",
            recognizer=recognizer,
            router=router,
            prompter=prompter,
            backfill=backfill,
            agent_b_callable=agent_b,
            image_root=args.image_root,
        )

    print(f"\n{SEPARATOR}")
    print("冒烟测试完成！")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
