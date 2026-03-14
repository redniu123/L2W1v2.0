Set-Location G:\Code\PaddleOCR\L2W1

$train = (Get-Content data\raw\hctr_riskbench\train.jsonl | Measure-Object -Line).Lines
$val   = (Get-Content data\raw\hctr_riskbench\val.jsonl   | Measure-Object -Line).Lines
$test  = (Get-Content data\raw\hctr_riskbench\test.jsonl  | Measure-Object -Line).Lines
Write-Host "Train: $train / Val: $val / Test: $test"

G:\envs\l2w1v2\python.exe scripts\test_efficiency_100.py `
  --test_jsonl data/raw/hctr_riskbench/test.jsonl `
  --image_root data/geo `
  --rec_model_dir models/agent_a_ppocr/PP-OCRv5_server_rec_infer `
  --rec_char_dict_path ppocr/utils/ppocrv5_dict.txt `
  --geo_dict data/dicts/Geology.txt `
  --output_dir results/stage2_v51 `
  --n_samples 100
