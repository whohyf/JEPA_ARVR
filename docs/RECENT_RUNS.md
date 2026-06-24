# Recent HPC Runs — Entry Point Reference

更新日期：2026-06-24

这份文档列出 VJEPA2-EXP 近期（2026-06-14 ~ 06-24）HPC 实验的入口脚本和启动命令，方便在共享仓库中对照代码和配置。完整指标和实验历史去 VJEPA2-EXP `logs/DASHBOARD.md` 和 `logs/RUNNING.md`。

> **注意：** 本仓库中的脚本路径（`/path/to/VJEPA2-EXP`）和 slurm account 已 sanitize。实际提交时通过 `PROJECT_ROOT`、`SHARED_PROJECT_ROOT` 等环境变量覆盖，或用 submit wrapper 自动检测项目根目录。

> **Comparability:** 只有 `p01_fixed` split + `phd_reference` class-space/anchor 的 run 之间可直接对比。`legacy_*` run 使用旧划分/旧语义，只用于复现历史 baseline。

---

## 当前标准（2026-06-15 起）

| 项 | 值 |
|---|---|
| Backbone | ViT-L/256 |
| Split | `p01_fixed` (train 5964 / val 375 / test 1045) |
| Class-space | `phd_reference` |
| Temporal | `phd_reference` (action-start anchor) |
| Probe | singleprobe |
| Precision | fp32 |
| GPU | H100 80GB |

---

## p01_fixed / PhD-Reference 对齐 Run

### 1. RGB-only Probe-only Baseline（p01_fixed，冻结编码器）

| 项 | 值 |
|---|---|
| lora-tag | `hdepic-singleprobe-1s-p01fixed-rgbonly-probeonly-vitl-fp32-bs8-noac-10ep-w16` |
| Job ID | 10993397 |
| 提交脚本 | `scripts/submit_b11_singleprobe_1s_p01fixed_rgbonly_probeonly.sh` |
| Config | `configs/generated/hdepic_singleprobe_1s_p01fixed_rgbonly_probeonly_l40s_vitl_fp32_bs8_noac_fulltrain.yaml` |
| 说明 | 冻结编码器，只训练 probe。p01_fixed split。相当于参考 `LORA_RANK=0` baseline。 |

### 2. RGB-only Encoder-LoRA Baseline（p01_fixed）

| 项 | 值 |
|---|---|
| lora-tag | `hdepic-singleprobe-1s-p01fixed-rgbonly-encoderlora-vitl-fp32-bs8-noac-10ep-w16` |
| Job ID | 10992528 |
| 提交脚本 | `scripts/submit_b11_singleprobe_1s_p01fixed_rgbonly_encoderlora_resumable.sh`（位于 worktree `encoderlora-p01fixed-rgbonly-baseline`） |
| Config | `configs/generated/hdepic_singleprobe_1s_p01fixed_rgbonly_encoderlora_vitl_fp32_bs8_noac_fulltrain.yaml` |
| 说明 | encoder-LoRA (rank=8, alpha=16, attn.qkv+attn.proj, all blocks)。Encoder-LoRA 参数与 JEPA_ARVR 对齐。 |

### 3. RGB-only Probe-only Baseline（p01_fixed，L40S 并行副本）

| 项 | 值 |
|---|---|
| lora-tag | `hdepic-singleprobe-1s-p01fixed-rgbonly-probeonly-vitl-fp32-bs8-noac-10ep-l40s` |
| Job ID | 11040527（chain，多次 relaunch） |
| 提交脚本 | `scripts/submit_b11_singleprobe_1s_p01fixed_rgbonly_probeonly_l40s.sh` |
| 说明 | H100 p01_fixed probe-only baseline 的 L40S 副本。非紧急 LTM 推理 baseline。 |

---

## VLM 实验（Visual Language Model — 替代 V-JEPA 编码器）

VLM 实验使用 HuggingFace 视觉/多模态模型的 frozen encoder 替代 V-JEPA2 作为特征提取器，与 V-JEPA 系列 **不在同一特征空间**，指标不可直接对比。所有 VLM 实验均使用 `p01_fixed` split。

### 零样本 / 少样本直接 Prompting（不训练）

直接用 VLM 做 action anticipation，不做任何微调。支持 zero-shot 和 few-shot（从 train set 采样示例注入 prompt）。

```bash
# 以 Llama-3.2-Vision 做零样本推理（test split）
BACKEND=llama32vision bash scripts/submit_vlm_zeroshot_prompting.sh

# LLaVA-OneVision 零样本
BACKEND=llava_onevision bash scripts/submit_vlm_zeroshot_prompting.sh

# 少样本（5-shot, test split）
BACKEND=llama32vision FEW_SHOT_K=5 bash scripts/submit_vlm_zeroshot_prompting.sh

# Smoke test（仅 5 样本，30 分钟时限）
BACKEND=llama32vision MAX_SAMPLES=5 SLURM_TIME=00:30:00 bash scripts/submit_vlm_zeroshot_prompting.sh
```

| 入口脚本 | Slurm 脚本 | Python 入口 |
|---|---|---|
| `scripts/submit_vlm_zeroshot_prompting.sh` | `scripts/run_vlm_zeroshot_prompting.slurm` | `app/hdepic_lora_action_anticipation/zeroshot_vlm_prompting.py` |

关键参数：`BACKEND`（模型族）、`NUM_FRAMES`（默认 32）、`MAX_SAMPLES`（0=全量）、`FEW_SHOT_K`（0=zero-shot）、`MAX_NEW_TOKENS`（默认 32）。

---

### Frozen-VLM-as-Encoder Probe Baseline（冻结 VLM，仅训练 probe，**不加任何 LoRA**）

用冻结的 HF VLM 视觉编码器提取特征，然后训练 attentive classification head（probe）。**VLM encoder 完全冻结** — `ENCODER_LORA_ENABLED=0`, `PREDICTOR_LORA_ENABLED=0`，不做任何 LoRA 微调，仅通过 `LORA_PROBE_TRAIN_MODE=full` 训练 probe。流程与 V-JEPA probe-only baseline 一致，但编码器替换为 `MODEL_FAMILY=vlm`。

> **与 refer_repo 的关系：** `refer_repo/JEPA_ARVR`（commit `add6d34`）包含 encoder-LoRA 方案但针对 V-JEPA 编码器，不涉及 VLM。VLM baseline 是本项目的独立扩展。

```bash
# CLIP-L/14@224 8帧 pooled token
VLM_MODEL_ID=openai/clip-vit-large-patch14 VLM_MODEL_CLASS=clip \
  bash scripts/submit_b11_vlm_baseline_1s_p01fixed_rgbonly_probeonly.sh

# Llama-3.2-Vision-11B 8帧
VLM_MODEL_ID=meta-llama/Llama-3.2-11B-Vision VLM_MODEL_CLASS=mllama \
  bash scripts/submit_b11_vlm_baseline_1s_p01fixed_rgbonly_probeonly_llama32vision.sh

# Resume from checkpoint
bash scripts/submit_b11_vlm_baseline_1s_p01fixed_rgbonly_probeonly_resumable.sh
```

| 项 | 值 |
|---|---|
| 提交脚本 | `scripts/submit_b11_vlm_baseline_1s_p01fixed_rgbonly_probeonly.sh`（CLIP）/ `..._llama32vision.sh`（Mllama） |
| Slurm 脚本 | `scripts/run_hdepic_lora_probe.slurm`（通用 probe 训练脚本，`MODEL_FAMILY=vlm`） |
| Config | `configs/generated/hdepic_vlm_baseline_1s_p01fixed_rgbonly_probeonly.yaml` / `..._llama32vision.yaml` |
| Python 入口 | `eval.py`（经过 `vlm_video_encoder.py` 获取视觉特征 → `eval_probe_checkpoint.py` 计算最终指标） |
| 说明 | 冻结 VLM encoder，`LORA_PROBE_TRAIN_MODE=full` 训练 attentive probe。H100 1 GPU，~256GB mem，batch=32，10 epochs。 |

---

### VLM LoRA SFT（语言模型 LoRA 微调）

对 LLaVA-OneVision 的 language model 做 LoRA SFT，用 HD-EPIC 动作标签作为生成目标。不经过 probe head — 直接让 LLM 输出 (verb, noun) 标签。

```bash
# 完整训练（1 epoch, LoRA rank=16）
bash scripts/submit_vlm_lora_sft.sh

# Smoke test（20 样本）
MAX_TRAIN_SAMPLES=20 SAVE_EVERY_STEPS=1 SLURM_TIME=00:30:00 bash scripts/submit_vlm_lora_sft.sh
```

| 入口脚本 | Slurm 脚本 | Python 入口 |
|---|---|---|
| `scripts/submit_vlm_lora_sft.sh` | `scripts/run_vlm_lora_sft.slurm` | `app/hdepic_lora_action_anticipation/train_vlm_lora_sft.py` |

关键参数：`NUM_EPOCHS`（默认 1）、`LR`（默认 1e-4）、`LORA_R`（默认 16）、`GRAD_ACCUM_STEPS`（默认 8）、`NUM_FRAMES`（默认 32）。

---

### VLM LoRA + Classification-Head Probe（LoRA + 分类头）

参考 PhD 的 Qwen2.5-VL-3B probe 方案：对 VLM 做 LoRA 微调，同时在视觉 token 上接 attentive classification head。支持多个后端。

```bash
# Qwen2.5-VL-3B
BACKEND=qwen25vl bash scripts/submit_vlm_probe_lora.sh

# LLaVA-OneVision
BACKEND=llava_onevision bash scripts/submit_vlm_probe_lora.sh

# Llama-3.2-Vision
BACKEND=llama32vision bash scripts/submit_vlm_probe_lora.sh

# Smoke test（16 样本, 1 epoch）
BACKEND=llava_onevision MAX_TRAIN_SAMPLES=16 NUM_EPOCHS=1 SLURM_TIME=00:30:00 RUN_TAG=smoke bash scripts/submit_vlm_probe_lora.sh
```

| 入口脚本 | Slurm 脚本 | Python 入口 |
|---|---|---|
| `scripts/submit_vlm_probe_lora.sh` | `scripts/run_vlm_probe_lora.slurm` | `app/hdepic_lora_action_anticipation/train_vlm_probe_lora.py` |

关键参数：`BACKEND`（必须，qwen25vl / llava_onevision / llama32vision）、`NUM_FRAMES`（默认 32）、`PROBE_NUM_FRAMES`（默认 8）、`LORA_RANK`（默认 16）、`BATCH_SIZE`（默认 4）、`GRAD_ACCUM_STEPS`（默认 2）、`NUM_EPOCHS`（默认 10）。

---

### Standalone Checkpoint 评估

对已训好的 probe checkpoint 做独立测试集评估（不重新训练）：

```bash
# VLM probe checkpoint eval
python app/hdepic_lora_action_anticipation/eval_probe_checkpoint.py \
  --checkpoint /path/to/checkpoint.pt \
  --config /path/to/config.yaml \
  --split test
```

| Python 入口 | 说明 |
|---|---|
| `app/hdepic_lora_action_anticipation/eval_probe_checkpoint.py` | 独立 eval，支持 VLM 和 V-JEPA 两种 checkpoint |

---

## Legacy 语义 Run（有意分歧，仅用于复现历史）

### 4. Legacy RGB-only Encoder-LoRA（legacy split）

| 项 | 值 |
|---|---|
| lora-tag | `hdepic-singleprobe-1s-legacy-rgbonly-vitl-fp32-bs8-noac-10ep-w10` |
| Job ID | 10910499 |
| 提交脚本 | `scripts/submit_b11_singleprobe_1s_legacy_rgbonly.sh` |
| 说明 | legacy split (5744/1397)，显式设 `LORA_TEMPORAL_SAMPLING=legacy` + `LORA_CLASS_SPACE=train_only`。复现历史 job 10847438 的旧语义 baseline。 |

### 5. Legacy Gaze+No-Pose（legacy split）

| 项 | 值 |
|---|---|
| lora-tag | `hdepic-singleprobe-1s-legacy-gaze-nopose-vitl-fp32-bs8-noac-10ep-w4` |
| Job ID | 10847438 |
| 提交脚本 | `scripts/submit_b11_singleprobe_1s_legacy_gaze_nopose.sh` |
| 说明 | gaze-only, no pose。encoder-LoRA + binary_input_adapter_gaze_pose_matrix (gaze channel only)。 |

### 6. Legacy Gaze+Pose+Reg（legacy split，含 confidence penalty）

| 项 | 值 |
|---|---|
| lora-tag | `hdepic-singleprobe-1s-legacy-gazepose-regcp-w0p01-vitl-fp32-bs8-noac-10ep-w4` |
| Job ID | 10997374 |
| 提交脚本 | `scripts/submit_b11_singleprobe_1s_legacy_gazepose_reg_vitl_fp32_bs8_noac_fulltrain.sh` |
| 说明 | gaze+pose+confidence_penalty=0.01。⚠️ DC-002: legacy split 的 test CSV 100% 包含在 train 中（data leak），test metrics 无意义。 |

### 7. Legacy RGB-only Predictor-LoRA（legacy split）

**核心差异：** encoder 完全冻结（`ENCODER_LORA_ENABLED=0`），LoRA 注入 predictor 的全部 12 blocks 的 `attn.qkv` + `attn.proj`（`PREDICTOR_LORA_ENABLED=1`）。与 encoder-LoRA baseline（#4，job 10910499）使用完全相同的 split / 数据 / 优化超参 / 资源，仅 LoRA 目标不同，因此指标差异可直接归因于 LoRA placement。

```bash
# 完整训练
bash scripts/submit_b11_singleprobe_1s_legacy_rgbonly_predictorlora_vitl_fp32_bs8_noac_fulltrain.sh

# Resume from checkpoint
bash scripts/submit_b11_singleprobe_1s_legacy_rgbonly_predictorlora_resumable.sh
```

| 项 | 值 |
|---|---|
| lora-tag | `hdepic-singleprobe-1s-legacy-rgbonly-predictorlora-vitl-fp32-bs8-noac-10ep-w10` |
| Job ID | 11094405（第 6 次 attempt，resume from ep0） |
| 提交脚本 | `scripts/submit_b11_singleprobe_1s_legacy_rgbonly_predictorlora_vitl_fp32_bs8_noac_fulltrain.sh` |
| Slurm 脚本 | `scripts/run_hdepic_lora_probe.slurm`（通用 probe 训练脚本） |
| Config | `configs/generated/hdepic_singleprobe_1s_legacy_rgbonly_predictorlora_vitl_fp32_bs8_noac_fulltrain.yaml` |
| 参数 | `ENCODER_LORA_ENABLED=0`, `PREDICTOR_LORA_ENABLED=1`, `PREDICTOR_LORA_RANK=8`, `PREDICTOR_LORA_ALPHA=16.0`, `PREDICTOR_LORA_LR_MULT=0.5`, `PREDICTOR_LORA_TARGET_SUFFIXES=attn.qkv\|attn.proj`, `LORA_CLASS_SPACE=train_only`, `LORA_TEMPORAL_SAMPLING=legacy` |
| 说明 | encoder 完全冻结，predictor 全部 12 blocks 注入 LoRA。项目独有功能（refer_repo 只有 encoder-LoRA，无 predictor-LoRA）。256GB mem, 16 cpus, batch=8。 |

---

## AR 10s Run

### 8. AR10s Direct-Rope（legacy split）

| 项 | 值 |
|---|---|
| lora-tag | `hdepic-singleprobe-ar10s-direct-rope-vitl-fp32-bs8-noac-10ep-w4-tr8-10` |
| Job ID | 10825859 |
| Config | `configs/generated/hdepic_singleprobe_ar10s_direct_rope_gaze_pose_enclora_fulltrain.yaml` |
| 说明 | direct_rope 10s forward path。⚠️ legacy split，predates `8f1adea` 对齐修复。 |

### 9. AR10s Sliding Window（legacy split，resource probe）

| lora-tag | Job ID | 说明 |
|---|---|---|
| `hdepic-ar10s-sliding3-bs8-probe-i15` | 10828617 | 3-step AR, bs=8 |
| `hdepic-ar10s-sliding3-bs6-probe-i15` | 10828618 | 3-step AR, bs=6 |
| `hdepic-ar10s-sliding3-bs4-probe-i15` | 10828619 | 3-step AR, bs=4 |
| `hdepic-ar10s-sliding3-bs2-probe-i15` | 10828620 | 3-step AR, bs=2 |

---

## B11 Interframe Pose Matrix（ViT-G，pre-ViT-L）

### 10. B11 Matrix Full Train（ViT-G/384，legacy split）

| 项 | 值 |
|---|---|
| lora-tag | `hdepic-lora-binary-gaze-pose-matrix-5ep-bs2-w10-m768-p01` |
| Job ID | 10024784 (train), 10081063 (formal val) |
| Config | `configs/generated/hdepic_lora_binary_input_adapter_gaze_pose_matrix.yaml` |
| 说明 | ViT-G/384, B5 5ch adapter + interframe pose matrix。⚠️ pre-ViT-L，指标不应直接与 ViT-L run 对比。 |

---

## D3 / LTM 诊断 Run

| 实验 | Script | 说明 |
|---|---|---|
| D3-E11 latent distribution @10s | `scripts/run_d3_e11_latent_distribution_10s.slurm` | 10s horizon latent 分布分析 |
| D3-E12 norm rescale @10s | `scripts/run_d3_e12_norm_rescale_10s.slurm` | latent norm rescale 实验 |
| D3-E14 ctx0 prediction dist @10s | `scripts/run_d3_e14_ctx0_prediction_dist_10s.slurm` | ctx=0 时 predictor 输出分布 |
| D3-E17 junk future @10s | `scripts/run_d3_e17_junk_future_10s.slurm` | junk future token 控制实验 |
| D3-E18a direct probe train | `scripts/run_d3_e18a_direct_probe_train.slurm` | 直接 probe 训练 |
| D3-E18a direct valonly | `scripts/run_d3_e18a_direct_valonly.slurm` | 直接 probe val-only |
| D3-E20 oracle probe train | `scripts/run_d3_e20_oracle_probe_train.slurm` | oracle future probe 训练 |

---

## Encoder-LoRA + Single-Probe（Gaze+Pose，B11 对齐）

### Encoder-LoRA 20head LowLR（H100，完整 gaze+pose matrix）

| 项 | 值 |
|---|---|
| lora-tag | `hdepic-20head-lora-enclora-gaze-pose-lrscale002-r4-last4-bs2-10ep` |
| 提交脚本 | `scripts/submit_b11_enclora_20head_lowlr_fulltrain.sh` |
| Slurm 脚本 | `scripts/run_hdepic_encoder_lora_gaze_pose_matrix_20head_lowlr_h100.slurm` |
| Config | `configs/generated/hdepic_lora_encoder_lora_gaze_pose_matrix_20head_lowlr_fulltrain.yaml` |
| Smoke 脚本 | `scripts/submit_b11_enclora_20head_lowlr_fullsmoke.sh` |
| 说明 | Encoder-LoRA rank=4 最后 4 blocks，probe LR scale=0.02（整体 LR 降低）。20 probe heads。 |

### Single-Probe Encoder-LoRA Gaze+Pose Matrix（简化 probe）

| 项 | 值 |
|---|---|
| lora-tag | `hdepic-singleprobe-new-full-enclora-gaze-pose-h100-r8-allblocks-bs4-10ep` |
| 提交脚本 | `scripts/submit_b11_singleprobe_new_enclora_fulltrain.sh` |
| Slurm 脚本 | `scripts/run_hdepic_single_probe_encoder_lora_gaze_pose_matrix_h100.slurm` |
| Config (fulltrain) | `configs/generated/hdepic_single_probe_encoder_lora_gaze_pose_matrix_fulltrain.yaml` |
| Config (smoke) | `configs/generated/hdepic_single_probe_encoder_lora_gaze_pose_matrix_smoke.yaml` |
| 说明 | Single probe, encoder-LoRA rank=8 all blocks。`LORA_PROBE_TRAIN_MODE=full`。batch=4, 10 epochs。 |

---

## AR 10s Sliding Gaze+Pose EncoRA

| 项 | 值 |
|---|---|
| lora-tag | `hdepic-singleprobe-ar10s-sliding-fp32-enclora-gaze-pose-h100-bs4-10ep-w2` |
| 提交脚本 | `scripts/submit_b11_singleprobe_ar10s_sliding_gaze_pose_fulltrain.sh` |
| Smoke 脚本 | `scripts/submit_b11_singleprobe_ar10s_sliding_gaze_pose_ddp_smoke.sh` |
| Slurm 脚本 | `scripts/run_hdepic_single_probe_encoder_lora_gaze_pose_matrix_h100.slurm` |
| Config | `configs/generated/hdepic_singleprobe_ar10s_sliding_gaze_pose_enclora_fulltrain.yaml` |
| 说明 | 10s AR sliding window + gaze/pose matrix + encoder-LoRA。`HDEPIC_AR10S_SLIDING_GAZE_POSE=1`, `EVAL_ANTICIPATION_SEC=10`, `MODEL_RETURN_MODE=final_window`。val every 2 epochs。 |

---

## 对照指南

1. **对齐参考实现（V-JEPA）** → 看 p01_fixed run（#1, #2）
2. **VLM 系列实验** → 看 VLM 实验章节：零样本 prompting、frozen-VLM probe（不加 LoRA）、LoRA SFT、LoRA+probe
3. **复现历史 baseline** → 看 legacy run（#4, #5）
4. **验证 predictor-LoRA** → 看 legacy predictor-LoRA（#7），启动命令见该节
5. **Encoder-LoRA + gaze/pose** → 看 Encoder-LoRA 20head LowLR / Single-Probe 章节
6. **AR long-horizon + gaze/pose** → 看 AR 10s Sliding Gaze+Pose 章节
7. **Latent 诊断** → 看 D3/LTM 系列
8. **Standalone eval（已训模型）** → `python app/hdepic_lora_action_anticipation/eval_probe_checkpoint.py --checkpoint <path> --config <path> --split test`

### LoRA 方案对照

| 实验 | Encoder LoRA | Predictor LoRA | Probe 训练 | 备注 |
|---|---|---|---|---|
| p01_fixed RGB-only probe-only | ❌ | ❌ | full | 冻结编码器 baseline |
| p01_fixed Encoder-LoRA | ✅ rank=8 | ❌ | full | all blocks, attn.qkv+proj |
| Legacy Encoder-LoRA (#4) | ✅ rank=8 | ❌ | full | legacy split |
| **Legacy Predictor-LoRA (#7)** | ❌ | ✅ rank=8 | full | encoder 冻结，predictor all blocks attn.qkv+proj |
| EncoRA 20head LowLR | ✅ rank=4 last 4 | ❌ | full | 降低 probe LR |
| **VLM frozen probe** | ❌ | ❌ | full | VLM encoder 完全冻结 |
| VLM LoRA + Probe | ✅ (VLM LoRA) | ❌ | full | HF PEFT，非 V-JEPA |
| `refer_repo/JEPA_ARVR` | ✅ | ❌ | — | 参考 encoder-LoRA，无 predictor-LoRA/无 VLM |

完整 run 注册表（含 checkpoint 路径、sidecar、raw log 链接）去 VJEPA2-EXP `logs/RUNNING.md`。
