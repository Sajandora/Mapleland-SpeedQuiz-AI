#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_or_infer_resnet.py – ResNet-50(PyTorch)
────────────────────────────────────────────────────────
• 모델(pt)이 없으면 학습, 있으면 곧바로 추론
• AMP(fp16) · MixUp(on/off) · Windows multiproc 안전
• 512 클래스 / ResNet-50 224×224 해상도
"""

import os, sys, numpy as np, torch, timm
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from functools import partial
from torch.amp      import autocast
from torch.cuda.amp import GradScaler

# ───── 하이퍼파라미터 ────────────────────────────────────
IMG_SIZE, BATCH     = 224, 32
E1,  E2             = 10, 30
UNFREEZE            = 60
LR_HEAD, LR_FINE    = 1e-3, 5e-5
MIXUP_A             = 0.1
USE_MIXUP           = True
PATIENCE, SEED      = 6, 42
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(">>> DEVICE =", DEVICE)

# ───── Dataset 정의 ─────────────────────────────────────
class SpriteDS(Dataset):
    def __init__(self, samples, tf):
        self.s, self.tf = samples, tf
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        p, l = self.s[i]
        return self.tf(Image.open(p).convert("RGB")), l

def mixup_collate(batch, n_cls, alpha):
    x, y = list(zip(*batch))
    x = torch.stack(x)
    y = torch.nn.functional.one_hot(torch.tensor(y), n_cls).float()
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    return lam*x + (1-lam)*x[idx], lam*y + (1-lam)*y[idx]

def default_collate(batch):
    x, y = list(zip(*batch))
    return torch.stack(x), torch.tensor(y)

def accuracy(pred, target):
    if target.dim() == 2:
        target = target.argmax(1)
    return (pred.argmax(1) == target).float().mean().item()

class EarlyStop:
    def __init__(s, p): s.p, s.best, s.cnt = p, 1e9, 0
    def step(s, v):
        s.cnt = 0 if v < s.best else s.cnt + 1
        s.best = min(s.best, v)
        return s.cnt >= s.p

def soft_ce(logits, soft_targets):
    log_prob = torch.log_softmax(logits, dim=1)
    return (-soft_targets * log_prob).sum(dim=1).mean()

# ───── main ─────────────────────────────────────────────
def main():
    BASE  = os.path.dirname(os.path.abspath(__file__))
    DATA  = os.path.join(BASE, "util/img")
    SAVE  = os.path.join(BASE, "converted_savedmodel_resnet50")
    os.makedirs(SAVE, exist_ok=True)
    MODEL = os.path.join(SAVE, "model.pt")
    LABEL = os.path.join(SAVE, "labels.txt")

    # 1) 파일 스캔 --------------------------------------------------
    cls_dirs = sorted([d for d in os.listdir(DATA)
                       if d.isdigit() and os.path.isdir(os.path.join(DATA, d))],
                      key=lambda s: int(s))
    paths, labels = [], []
    for idx, cls in enumerate(cls_dirs):
        for f in os.listdir(os.path.join(DATA, cls)):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
                paths.append(os.path.join(DATA, cls, f)); labels.append(idx)
    n_cls = len(cls_dirs)
    print(f"총 {len(paths)} 장 · 클래스 {n_cls} 개")

    # 2) train / val split -----------------------------------------
    tr_idx, va_idx = next(StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=SEED).split(paths, labels))
    tr_s = [(paths[i], labels[i]) for i in tr_idx]
    va_s = [(paths[i], labels[i]) for i in va_idx]

    # 3) Transform & Loader ----------------------------------------
    base_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)])

    aug_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.3, 1.0)),
        transforms.RandomAffine(degrees=15, scale=(0.5, 1.0)),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)])

    tr_ld = DataLoader(
        SpriteDS(tr_s, aug_tf), BATCH, True, num_workers=4, pin_memory=True,
        collate_fn=partial(mixup_collate, n_cls=n_cls, alpha=MIXUP_A) if USE_MIXUP else default_collate)
    va_ld = DataLoader(
        SpriteDS(va_s, base_tf), BATCH, False, num_workers=2, pin_memory=True,
        collate_fn=default_collate)

    # 4) 모델 -------------------------------------------------------
    model = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=n_cls).to(DEVICE)
    criterion_int = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()

    def run(loader, train, opt=None):
        model.train(train)
        tot_l = tot_a = n = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if USE_MIXUP:
                y_soft = y if y.dim() == 2 else torch.nn.functional.one_hot(y, n_cls).float()
            else:
                y_soft = torch.nn.functional.one_hot(y, n_cls).float()

            with autocast(device_type="cuda"):
                out  = model(x)
                loss = soft_ce(out, y_soft) if USE_MIXUP else criterion_int(out, y)

            if train:
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()

            with torch.no_grad():
                tot_l += loss.item() * x.size(0)
                tgt = y.argmax(1) if y.dim() == 2 else y
                tot_a += accuracy(out, tgt) * x.size(0)
                n += x.size(0)
        return tot_l / n, tot_a / n

    # 5) 학습 or 추론 ----------------------------------------------
    if not os.path.exists(MODEL):
        print("\n[모델 없음] 학습을 시작합니다…")

        # Stage-1 : Head
        for p in model.parameters(): p.requires_grad = False
        for p in model.get_classifier().parameters(): p.requires_grad = True
        opt_h = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), LR_HEAD)
        sch_h = optim.lr_scheduler.ReduceLROnPlateau(opt_h, 'min', factor=0.5, patience=3, min_lr=1e-6)
        es = EarlyStop(PATIENCE);  print("\n[Stage-1] Head …")
        for e in range(1, E1+1):
            tr_l,tr_a = run(tr_ld, True,  opt_h)
            va_l,va_a = run(va_ld, False)
            sch_h.step(va_l)
            print(f"Ep{e:02d}/{E1}  tr {tr_l:.4f}/{tr_a:.3f}  val {va_l:.4f}/{va_a:.3f}")
            if es.step(va_l): break

        # Stage-2 : Fine-tune
        unf = 0
        for layer in reversed(list(model.children())):
            for p in layer.parameters():
                if unf < UNFREEZE:
                    p.requires_grad = True; unf += 1
        opt_f = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), LR_FINE)
        sch_f = optim.lr_scheduler.ReduceLROnPlateau(opt_f, 'min', factor=0.5, patience=3, min_lr=1e-6)
        es = EarlyStop(PATIENCE);  print(f"\n[Stage-2] Fine-tune last {unf} params …")
        for e in range(1, E2+1):
            tr_l,tr_a = run(tr_ld, True,  opt_f)
            va_l,va_a = run(va_ld, False)
            sch_f.step(va_l)
            print(f"Ep{e:02d}/{E2}  tr {tr_l:.4f}/{tr_a:.3f}  val {va_l:.4f}/{va_a:.3f}")
            if es.step(va_l): break

        torch.save(model.state_dict(), MODEL)
        with open(LABEL, "w", encoding="utf-8") as f:
            f.write("\n".join(cls_dirs))
        print("\n✅ 학습 완료!  →", MODEL)

    else:
        model.load_state_dict(torch.load(MODEL, map_location=DEVICE))
        model.eval()
        idx2cls = [ln.strip() for ln in open(LABEL, encoding="utf-8")]

        img_path = input("\n추론할 이미지 경로: ").strip()
        if not os.path.exists(img_path):
            print("❌ 이미지가 없습니다."); sys.exit()

        tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)])
        x = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)

        with torch.no_grad(), autocast(device_type="cuda"):
            pred = model(x)
        print(f"\n✅ 예측 결과: {idx2cls[pred.argmax(1).item()]}")

# Windows 진입점 ---------------------------------------------------
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
