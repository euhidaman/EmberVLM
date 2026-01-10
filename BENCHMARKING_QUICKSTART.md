# 🚀 Quick Start: Install VLMEvalKit for Benchmarking

## ⚡ Fast Setup (2 minutes)

```powershell
# 1. Run the setup script
python setup_vlmeval.py

# 2. Start training with benchmarking
$env:HF_TOKEN = "your_hf_token_here"
torchrun --nproc_per_node=2 scripts/train_all.py `
  --size small `
  --run_benchmarks `
  --benchmark_preset quick `
  --stage all `
  --distributed `
  --mixed_precision bf16
```

## 📦 What Gets Installed?

1. **VLMEvalKit** - Benchmark evaluation framework
2. **Dependencies** - openpyxl, apted, colormath, distance, decord

**Total size**: ~500 MB (including benchmark data on first run)

## 🎯 Training Without Benchmarking

If you don't want to install VLMEvalKit right now:

```powershell
# Simply don't use --run_benchmarks flag (benchmarking is opt-in)
torchrun --nproc_per_node=2 scripts/train_all.py `
  --size small `
  --stage all `
  --distributed `
  --mixed_precision bf16
```

## 📊 What Benchmarking Does

After Stage 2 (Instruction Tuning) completes, your model will be evaluated on:

- **MMBench** - General VLM understanding
- **TextVQA** - Text recognition in images  
- **ScienceQA** - Scientific reasoning
- **AI2D** - Diagram understanding
- **ChartQA** - Chart interpretation

Results are logged to WandB with comparison tables and visualizations.

## ⚙️ Benchmark Presets

- `--benchmark_preset quick` - 3 benchmarks, ~30 min ⚡
- `--benchmark_preset standard` - 5-6 benchmarks, ~1-2 hours (default) ✅
- `--benchmark_preset full` - 10+ benchmarks, ~4-6 hours 🔬

## 🛠️ Troubleshooting

### Issue: "VLMEvalKit not installed"
**Solution:** Run `python setup_vlmeval.py`

### Issue: Out of memory during benchmarking
**Solution:** Use `--benchmark_preset quick`

### Issue: Benchmarks taking too long
**Solution:** Run training overnight, or use `--skip_benchmarks`

## 📚 Full Documentation

See [VLMEVAL_SETUP.md](VLMEVAL_SETUP.md) for complete setup guide and advanced options.

## ✨ New Features Added

### Enhanced VLM Training:
- ✅ **Perplexity** metric for Stage 2 (language quality)
- ✅ **Top-5 accuracy & MRR** for Stage 1 (retrieval quality)  
- ✅ **Early stopping** with best checkpoint tracking
- ✅ **Increased epochs**: Stage 1: 3→7, Stage 2: 3→10

### Stage 2.5: Benchmark Evaluation:
- ✅ **Automated VLM benchmarking** using VLMEvalKit
- ✅ **Quality gating** - stops training if VLM is too weak
- ✅ **WandB logging** with comparison tables and charts
- ✅ **Graceful fallback** if VLMEvalKit not installed

### Training Visualizations:
- ✅ **visualize_training.py** script for plot generation
- ✅ **Stage summaries** with improvement statistics
- ✅ **Convergence tracking** for all stages

## 🎓 Updated Training Command

### With Benchmarking (Recommended):
```powershell
$env:HF_TOKEN = "hf_..."
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

torchrun --nproc_per_node=2 scripts/train_all.py `
  --size small `
  --run_benchmarks `
  --benchmark_preset standard `
  --quality_threshold auto `
  --stage all `
  --distributed `
  --mixed_precision bf16 `
  --batch_size 18 `
  --gradient_accumulation 16 `
  --stage1_epochs 7 `
  --stage2_epochs 10 `
  --stage3_robot_epochs 30 `
  2>&1 | Tee-Object -FilePath train.log
```

### Without Benchmarking (Faster):
```powershell
torchrun --nproc_per_node=2 scripts/train_all.py `
  --size small `
  --stage all `
  --distributed `
  --mixed_precision bf16 `
  --batch_size 18 `
  --gradient_accumulation 16 `
  2>&1 | Tee-Object -FilePath train.log
```

## 📈 Monitor Training

- **WandB Dashboard**: Real-time metrics, tables, and visualizations
- **Training Log**: `train.log` file with detailed progress
- **Checkpoints**: Saved in `outputs/` with best models marked

---

**Ready to train?** Run `python setup_vlmeval.py` and start benchmarking! 🎉
