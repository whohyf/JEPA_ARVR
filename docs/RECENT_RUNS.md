# Recent HPC Runs — Entry Point Reference

更新日期：2026-06-20

这份文档列出 VJEPA2-EXP 近期（2026-06-14 ~ 06-19）HPC 实验的入口脚本和 lora-tag，方便在共享仓库中对照代码和配置。完整指标和实验历史去 VJEPA2-EXP `logs/DASHBOARD.md` 和 `logs/RUNNING.md`。

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

| 项 | 值 |
|---|---|
| lora-tag | `hdepic-singleprobe-1s-legacy-rgbonly-predictorlora-vitl-fp32-bs8-noac-10ep-w10` |
| Job ID | 11094405（第 6 次 attempt，resume from ep0） |
| 提交脚本 | `scripts/submit_b11_singleprobe_1s_legacy_rgbonly_predictorlora_vitl_fp32_bs8_noac_fulltrain.sh` |
| 说明 | encoder 完全冻结，predictor 全部 12 blocks 注入 LoRA (rank=8, alpha=16, lr_mult=0.5)。项目独有功能，无参考对应。 |

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

## 对照指南

1. **对齐参考实现** → 看 p01_fixed run (#1, #2)
2. **复现历史 baseline** → 看 legacy run (#4, #5)
3. **验证 predictor-LoRA** → 看 legacy predictor-LoRA (#7)
4. **AR long-horizon** → 看 AR 10s (#8, #9)
5. **Latent 诊断** → 看 D3/LTM 系列

完整 run 注册表（含 checkpoint 路径、sidecar、raw log 链接）去 VJEPA2-EXP `logs/RUNNING.md`。
