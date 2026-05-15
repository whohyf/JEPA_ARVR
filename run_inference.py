"""
V-JEPA 2 真实推理脚本 (ViT-L + SSv2 attentive probe)
运行：
  cd /home/ll5914/ARVR_Video/vjepa2
  python ../run_inference.py
"""

import sys, os, time, json
sys.path.insert(0, "/home/ll5914/ARVR_Video/vjepa2")

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.attentive_pooler import AttentiveClassifier
from src.models.vision_transformer import vit_large_rope

MODEL_DIR    = "/scratch/ll5914/models/vjepa2"
ENCODER_CKPT = os.path.join(MODEL_DIR, "vitl.pt")
PROBE_CKPT   = os.path.join(MODEL_DIR, "ssv2-vitl-16x2x3.pt")
SSV2_CLASSES = os.path.join(MODEL_DIR, "ssv2_classes.json")
SAMPLE_VIDEO = os.path.join(MODEL_DIR, "sample_video.mp4")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

IMG_SIZE     = 256
NUM_FRAMES   = 16
FRAME_STEP   = 4
NUM_SEGMENTS = 2
NUM_VIEWS    = 3


def check_files():
    missing = []
    for path, name in [(ENCODER_CKPT, "vitl.pt"), (PROBE_CKPT, "ssv2 probe"),
                       (SSV2_CLASSES, "SSv2 classes"), (SAMPLE_VIDEO, "sample video")]:
        size = os.path.getsize(path) if os.path.exists(path) else -1
        if size <= 0:
            missing.append(f"  - {name}: {path}")
    if missing:
        print("[ERROR] 文件缺失：")
        for m in missing: print(m)
        sys.exit(1)
    for path, name in [(ENCODER_CKPT, "vitl.pt"), (PROBE_CKPT, "ssv2 probe")]:
        print(f"  [OK] {name}: {os.path.getsize(path)/1e6:.1f} MB")


def build_transform():
    short_side = int(256.0 / 224 * IMG_SIZE)
    return video_transforms.Compose([
        video_transforms.Resize(short_side, interpolation="bilinear"),
        video_transforms.CenterCrop(size=(IMG_SIZE, IMG_SIZE)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_video_clips(video_path):
    vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    total_frames = len(vr)
    print(f"  视频总帧数: {total_frames}, FPS: {vr.get_avg_fps():.1f}")
    clips = []
    clip_duration = NUM_FRAMES * FRAME_STEP
    starts = np.linspace(0, max(0, total_frames - clip_duration), NUM_SEGMENTS, dtype=int)
    for start in starts:
        indices = np.arange(start, start + clip_duration, FRAME_STEP, dtype=np.int64)
        indices = np.clip(indices, 0, total_frames - 1)
        frames = vr.get_batch(indices).asnumpy()
        for _ in range(NUM_VIEWS):
            clips.append(frames)
    return clips


def load_encoder(device):
    print("  加载 ViT-L encoder...")
    t0 = time.time()
    # vit_large_rope 内部已含 use_rope=True，不重复传
    model = vit_large_rope(
        img_size=(IMG_SIZE, IMG_SIZE),
        num_frames=NUM_FRAMES,
        tubelet_size=2,
        patch_size=16,
        uniform_power=True,
    )
    ckpt = torch.load(ENCODER_CKPT, map_location="cpu", weights_only=True)
    state = ckpt.get("target_encoder", ckpt.get("encoder", ckpt))
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    msg = model.load_state_dict(state, strict=False)
    print(f"  权重加载: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    model = model.to(device).eval()
    print(f"  完成，耗时 {time.time()-t0:.1f}s, embed_dim={model.embed_dim}")
    return model


def load_probe(embed_dim, device):
    print("  加载 SSv2 attentive probe...")
    t0 = time.time()
    probe = AttentiveClassifier(embed_dim=embed_dim, num_heads=16, depth=4, num_classes=174)
    ckpt = torch.load(PROBE_CKPT, map_location="cpu", weights_only=True)
    state = ckpt.get("classifiers", [ckpt])[0]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    msg = probe.load_state_dict(state, strict=False)
    print(f"  权重加载: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    probe = probe.to(device).eval()
    print(f"  完成，耗时 {time.time()-t0:.1f}s")
    return probe


def run_inference():
    print("=" * 60)
    print("V-JEPA 2 真实推理 — Something-Something v2 视频分类")
    print("=" * 60)

    check_files()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("\n[1] 加载模型")
    encoder = load_encoder(device)
    probe   = load_probe(encoder.embed_dim, device)

    with open(SSV2_CLASSES) as f:
        classes = json.load(f)

    print("\n[2] 读取视频")
    transform = build_transform()
    clips = load_video_clips(SAMPLE_VIDEO)
    print(f"  采样 {len(clips)} 个 clip ({NUM_SEGMENTS} segments × {NUM_VIEWS} views)，每 clip {NUM_FRAMES} 帧")

    print("\n[3] 推理")
    all_logits = []
    for i, clip in enumerate(clips):
        print(f"  clip {i+1}/{len(clips)} ...", end=" ", flush=True)
        t0 = time.time()
        frames = torch.from_numpy(clip).permute(0, 3, 1, 2)
        x = transform(frames).unsqueeze(0).to(device)
        with torch.no_grad():
            features = encoder(x)
            logits   = probe(features)
        all_logits.append(logits.cpu())
        print(f"{time.time()-t0:.1f}s")

    avg_logits = torch.stack(all_logits).mean(0)

    print("\n[4] 分类结果")
    print(f"  视频: {os.path.basename(SAMPLE_VIDEO)}")
    print()
    print("  Top-5 预测 (SSv2 动作类别):")
    probs = F.softmax(avg_logits, dim=-1)[0]
    top5 = probs.topk(5)
    for rank, (idx, prob) in enumerate(zip(top5.indices.tolist(), top5.values.tolist())):
        class_name = classes.get(str(idx), f"class_{idx}")
        bar = "█" * int(prob * 50)
        print(f"  #{rank+1}  {prob*100:5.2f}%  {bar:<25}  {class_name}")

    print("\n推理完成！")


if __name__ == "__main__":
    run_inference()
