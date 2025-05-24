import time, json, ctypes, numpy as np, pyautogui
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageOps, ImageTk
import torch
import timm
from torchvision import transforms
import os, sys

# 리소스 경로 설정 (PyInstaller 대응)
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# 경로 설정
MODEL_PATH = resource_path("converted_savedmodel_resnet50/model.pt")
LABELS_PATH = resource_path("converted_savedmodel_resnet50/labels.txt")
ANS_JSON_PATH = resource_path("answers.json")
try:
    TEST_IMAGE_PATH = resource_path("TEST6.PNG")
    if not os.path.exists(TEST_IMAGE_PATH):
        TEST_IMAGE_PATH = None
except Exception:
    TEST_IMAGE_PATH = None

CAPTURE_INTERVAL = 800
ROI_COLOR = "red"
ROI_WIDTH = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROI = None
OVERLAYS = []
RUNNING = False
DEBUG_MODE = False
TEST_MODE = False

labels = [ln.strip() for ln in open(LABELS_PATH, encoding="utf-8")]
num_classes = len(labels)

model = timm.create_model("resnet50", pretrained=False, num_classes=num_classes).to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

with open(ANS_JSON_PATH, encoding="utf-8") as f:
    ans_map = {item["img"]: item["answer"] for item in json.load(f)}

def id2answer(lbl: str) -> str:
    try:
        return ans_map.get(int(lbl), lbl)
    except ValueError:
        return lbl

def make_clickthrough(hwnd):
    GWL_EXSTYLE = -20
    WS_EX_LAYERED = 0x80000
    WS_EX_TRANSPARENT = 0x20
    style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    style |= WS_EX_LAYERED | WS_EX_TRANSPARENT
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)

def show_overlay():
    global OVERLAYS
    hide_overlay()
    if ROI is None:
        return
    x, y, w, h = ROI
    borders = []
    for bx, by, bw, bh in [
        (x, y, w, ROI_WIDTH),
        (x, y+h-ROI_WIDTH, w, ROI_WIDTH),
        (x, y, ROI_WIDTH, h),
        (x+w-ROI_WIDTH, y, ROI_WIDTH, h)
    ]:
        tl = tk.Toplevel()
        tl.overrideredirect(True)
        tl.geometry(f"{bw}x{bh}+{bx}+{by}")
        tl.configure(bg=ROI_COLOR)
        tl.attributes("-topmost", True)
        make_clickthrough(ctypes.windll.user32.GetParent(tl.winfo_id()))
        borders.append(tl)
    OVERLAYS = borders

def hide_overlay():
    global OVERLAYS
    for w in OVERLAYS:
        w.destroy()
    OVERLAYS = []

def select_roi_overlay():
    over = tk.Toplevel()
    over.attributes("-fullscreen", True)
    over.attributes("-alpha", 0.3)
    over.configure(bg="black")
    over.attributes("-topmost", True)
    cv = tk.Canvas(over, cursor="cross", bg="black")
    cv.pack(fill="both", expand=True)

    data = {"rect": None, "x": 0, "y": 0}
    def press(e):
        data["x"], data["y"] = e.x_root, e.y_root
        data["rect"] = cv.create_rectangle(data["x"], data["y"], data["x"], data["y"], outline="cyan", width=2)
    def drag(e):
        if data["rect"]:
            cv.coords(data["rect"], data["x"], data["y"], e.x_root, e.y_root)
    def release(e):
        global ROI
        x1, y1 = data["x"], data["y"]
        x2, y2 = e.x_root, e.y_root
        ROI = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        over.destroy()
        show_overlay()

    cv.bind("<ButtonPress-1>", press)
    cv.bind("<B1-Motion>", drag)
    cv.bind("<ButtonRelease-1>", release)
    over.mainloop()

def capture_roi():
    if ROI is None:
        raise RuntimeError("ROI가 정의되지 않았습니다.")
    x, y, w, h = ROI
    full = pyautogui.screenshot(region=(x, y, w, h))
    return full.crop((ROI_WIDTH, ROI_WIDTH, w - ROI_WIDTH, h - ROI_WIDTH))

@torch.no_grad()
def predict(img: Image.Image):
    img = ImageOps.fit(img, (300, 300), Image.Resampling.LANCZOS)
    x = tfm(img).unsqueeze(0).to(DEVICE)
    prob = torch.softmax(model(x), dim=1)[0]
    topk = torch.topk(prob, 6)
    top_indices = topk.indices.tolist()
    top_probs = topk.values.tolist()
    top_labels = [(id2answer(labels[i]), top_probs[idx]) for idx, i in enumerate(top_indices)]
    return top_labels[0], top_labels[1:6], img

def gui_app():
    root = tk.Tk()
    root.title("SpeedQuiz Solver (Overlay)")
    root.resizable(False, False)

    image_display = None
    result_labels = []

    def update_debug_results(results):
        for rl in result_labels:
            rl.destroy()
        result_labels.clear()
        for i, (lbl, conf) in enumerate(results):
            r_label = tk.Label(root, text=f"{lbl} ({conf:.2f})",
                               font=("맑은 고딕", 12),
                               fg="black", bg="#f0f0f0",
                               padx=6, pady=4,
                               cursor="hand2")
            r_label.pack(pady=2)
            def make_copy_callback(text, label=r_label):
                def on_click(e):
                    root.clipboard_clear()
                    root.clipboard_append(text)
                    label.config(fg="blue", bg="#d0f0ff")
                return on_click
            r_label.bind("<Button-1>", make_copy_callback(lbl))
            result_labels.append(r_label)

    def btn_set_roi():
        messagebox.showinfo("영역 지정", "드래그하여 ROI를 선택하세요.")
        select_roi_overlay()

    def loop():
        nonlocal image_display
        if not RUNNING:
            return
        try:
            if TEST_MODE:
                if not TEST_IMAGE_PATH:
                    raise FileNotFoundError("테스트 이미지(TEST6.PNG)가 존재하지 않습니다.")
                img_pil = Image.open(TEST_IMAGE_PATH).convert("RGB")
            else:
                img_pil = capture_roi()
            best, others, resized_img = predict(img_pil)
            ans, conf = best
            explanation = (
                "\n펠라스, 해리 둘이 똑같아서\n하나만 우직하게 그냥 하셈"
                if ans in ["펠라스", "해리"] else ""
            )

            if DEBUG_MODE:
                status.set(f"{ans}  ({conf:.2f}){explanation}\n\n그 외 추론 결과:")
                update_debug_results(others)
                tk_img = ImageTk.PhotoImage(resized_img)
                image_label.config(image=tk_img)
                image_label.image = tk_img
            else:
                image_label.config(image=None)
                image_label.image = None
                status.set(f"{ans}{explanation}")
                update_debug_results([])
        except Exception as e:
            status.set(f"⚠ {e}")
        root.after(CAPTURE_INTERVAL, loop)

    def start_live():
        global RUNNING, TEST_MODE
        if ROI is None:
            messagebox.showwarning("!", "ROI를 먼저 지정하세요.")
            return
        RUNNING, TEST_MODE = True, False
        loop()

    def start_test():
        global RUNNING, TEST_MODE
        if not DEBUG_MODE:
            messagebox.showinfo("디버그 모드 필요", "테스트 모드는 디버그 모드가 ON일 때만 사용 가능합니다.")
            return
        if not TEST_IMAGE_PATH:
            messagebox.showerror("오류", "TEST6.PNG 파일이 존재하지 않아 테스트를 실행할 수 없습니다.")
            return
        RUNNING, TEST_MODE = True, True
        loop()

    def stop_loop():
        global RUNNING
        RUNNING = False
        status.set("⏹ 중지됨")
        image_label.config(image=None)
        image_label.image = None
        update_debug_results([])

    def show_help():
        messagebox.showinfo("도움말", "1. '영역 지정' 버튼을 눌러 화면의 인식 영역을 드래그합니다.\n2. '실행'을 눌러 인식 시작. '중지'로 멈춥니다.\n3. '디버그 모드를 통해서 세부적인 결과를 얻어볼 수 있습니다. \n4. 인식이 잘 되지않으면 영역크기를 줄여주세요!")

    def toggle_debug():
        global DEBUG_MODE
        DEBUG_MODE = not DEBUG_MODE
        debug_btn.config(text=f"디버그 모드: {'ON' if DEBUG_MODE else 'OFF'}")

    top_frame = tk.Frame(root)
    top_frame.pack(fill="x", padx=5, pady=5)
    debug_btn = tk.Button(top_frame, text="디버그 모드: OFF", command=toggle_debug)
    debug_btn.pack(side="left")
    tk.Button(top_frame, text="❓ 도움말", command=show_help).pack(side="right")

    tk.Label(root, text="[영역 지정] ➜ [실행]").pack(pady=4)

    btn_fr = tk.Frame(root)
    btn_fr.pack(pady=6)
    tk.Button(btn_fr, text="영역 지정", width=11, command=btn_set_roi).grid(row=0, column=0, padx=3)
    tk.Button(btn_fr, text="실행", width=11, command=start_live).grid(row=0, column=1, padx=3)
    tk.Button(btn_fr, text="테스트", width=11, command=start_test).grid(row=0, column=2, padx=3)
    tk.Button(btn_fr, text="중지", width=11, command=stop_loop).grid(row=0, column=3, padx=3)

    status = tk.StringVar(value="대기 중")
    label = tk.Label(root, textvariable=status, font=("맑은 고딕", 14), cursor="hand2", justify="left")
    label.pack(pady=8)

    def copy_status_text(event):
        text = status.get().split('  ')[0]
        root.clipboard_clear()
        root.clipboard_append(text)
        label.config(fg="blue")
        root.after(1000, lambda: label.config(fg="black"))
        root.update()
    label.bind("<Button-1>", copy_status_text)

    image_label = tk.Label(root)
    image_label.pack(pady=4, side="bottom")

    root.mainloop()

if __name__ == "__main__":
    gui_app()
