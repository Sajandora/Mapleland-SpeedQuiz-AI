import os
import random
import numpy as np
from PIL import Image
import cv2

# 배경 색 정의
OLD_BG = (0, 0, 0)
NEW_BG = (60, 124, 193)
TOLERANCE = 10

# 배경 교체 함수
def replace_background(arr, old_rgb=OLD_BG, new_rgb=NEW_BG):
    mask = np.all(np.abs(arr[:, :, :3] - old_rgb) <= TOLERANCE, axis=-1)
    arr[mask] = list(new_rgb) + [255] if arr.shape[2] == 4 else list(new_rgb)
    return arr

# 증강 함수
def augment_image(img):
    augmentations = []
    flipped = np.fliplr(img)
    augmentations.append(flipped)

    angle = random.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))
    augmentations.append(rotated)

    brightness = random.uniform(0.7, 1.3)
    bright = np.clip(img * brightness, 0, 255).astype(np.uint8)
    augmentations.append(bright)

    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    noisy = cv2.add(img, noise)
    augmentations.append(noisy)

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    augmentations.append(blurred)

    small = cv2.resize(img, (w // 2, h // 2))
    resized = cv2.resize(small, (w, h))
    augmentations.append(resized)

    return augmentations

# 리사이즈 후 증강
def augment_image_resized(img):
    img_resized = cv2.resize(img, (224, 224))
    return augment_image(img_resized), img_resized

# 현재 경로 내 숫자 폴더 탐색
base_dir = os.getcwd()
folders = [f for f in os.listdir(base_dir) if f.isdigit() and os.path.isdir(os.path.join(base_dir, f))]
folders = sorted(folders, key=lambda x: int(x))

if not folders:
    print("❌ 숫자 폴더가 없습니다.")
    exit()

for folder in folders:
    print(f"\n📂 처리 중: {folder}")
    folder_path = os.path.join(base_dir, folder)

    images = [f for f in os.listdir(folder_path) if f.lower().endswith(".png") and "_aug" not in f and "_bg" not in f]
    if not images:
        print(f"⚠️ {folder} 폴더에 처리할 PNG가 없습니다.")
        continue

    for fname in images:
        img_path = os.path.join(folder_path, fname)
        base_name, ext = os.path.splitext(fname)

        try:
            img_np = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"⚠️ {fname} 읽기 오류: {e}")
            continue

        # ── 1. 원본 증강
        augs = augment_image(img_np)
        for i, aug in enumerate(augs, 1):
            out_path = os.path.join(folder_path, f"{base_name}_aug{i}{ext}")
            Image.fromarray(aug).save(out_path)

        # ── 2. 리사이즈 + 증강
        resize_augs, resized_img = augment_image_resized(img_np)
        resized_path = os.path.join(folder_path, f"{base_name}_resized{ext}")
        Image.fromarray(resized_img).save(resized_path)
        for i, aug in enumerate(resize_augs, 1):
            out_path = os.path.join(folder_path, f"{base_name}_resize_aug{i}{ext}")
            Image.fromarray(aug).save(out_path)

        # ── 3. 배경 교체 버전 (원본+증강 전체 대상)
        gen_images = [f for f in os.listdir(folder_path)
                      if f.startswith(base_name)
                      and f.endswith(ext)
                      and "_bg" not in f]

        for gname in gen_images:
            gpath = os.path.join(folder_path, gname)
            arr = np.array(Image.open(gpath).convert("RGBA"))
            arr_replaced = replace_background(arr.copy())
            bg_name = gname.replace(base_name, f"{base_name}_bg", 1)
            Image.fromarray(arr_replaced).save(os.path.join(folder_path, bg_name))

        print(f"✅ {fname} 완료")

print("\n🎉 전체 이미지 증강 + 배경 교체 완료!")
