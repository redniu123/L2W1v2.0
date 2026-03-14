Set-Location G:\Code\PaddleOCR\L2W1

G:\envs\l2w1v2\python.exe scripts\run_efficiency_frontier.py `
  --test_jsonl data/raw/hctr_riskbench/test.jsonl `
  --image_root data/geo `
  --rec_model_dir models/agent_a_ppocr/PP-OCRv5_server_rec_infer `
  --rec_char_dict_path ppocr/utils/ppocrv5_dict.txt `
  --geo_dict data/dicts/Geology.txt `
  --output_dir results/stage2_v51
