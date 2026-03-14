@echo off
echo ============================================================
echo SH-DA++ v5.1 Phase 0: 数据集切分 (Train 60%% / Val 20%% / Test 20%%)
echo ============================================================

cd /d G:\Code\PaddleOCR\L2W1

G:\envs\l2w1v2\python.exe scripts\split_dataset.py ^
    --input_jsonl data/geo/geotext.jsonl ^
    --output_dir data/raw/hctr_riskbench ^
    --train_ratio 0.6 ^
    --val_ratio 0.2 ^
    --test_ratio 0.2 ^
    --seed 42

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] 切分失败，请检查错误信息
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 切分完成！验证文件...
echo ============================================================
G:\envs\l2w1v2\python.exe -c "import json; train=sum(1 for _ in open('data/raw/hctr_riskbench/train.jsonl')); val=sum(1 for _ in open('data/raw/hctr_riskbench/val.jsonl')); test=sum(1 for _ in open('data/raw/hctr_riskbench/test.jsonl')); total=train+val+test; print(f'Train: {train} / Val: {val} / Test: {test} / Total: {total}')"

echo.
echo 下一步：运行 100 样本快速测试
echo   G:\envs\l2w1v2\python.exe scripts\test_efficiency_100.py --test_jsonl data/raw/hctr_riskbench/test.jsonl --image_root data/geo
echo.
pause
