<div align="center">
  <h1>⚡ 메이플랜드 스피드퀴즈 정답 예측기</h1>
  <p>딥러닝 기반 이미지 분류 모델로 실시간 정답 추론!</p>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch" />
  <img src="https://img.shields.io/badge/Tkinter-GUI-green" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
</div>

---

## 📌 프로젝트 소개

> **메이플랜드**에서 진행되는 **스피드퀴즈** 콘텐츠를 위한  
> **AI 기반 이미지 정답 예측 애플리케이션**입니다.  
> 화면 일부를 실시간으로 캡처하여 퀴즈 이미지를 분류하고,  
> 사전 학습된 AI 모델을 통해 정답을 예측하여 GUI에 표시합니다.

---

## 🧠 AI 모델 정보

- ✅ 모델 구조: `ResNet-50` (timm 라이브러리 기반)
- ✅ 프레임워크: `PyTorch`
- ✅ 입력 크기: 224 × 224

---

## 🎮 주요 기능

- 🖼️ ROI(영역) 지정 후 실시간 퀴즈 이미지 캡처
- ⚡ 예측 결과 GUI 표시 + 클릭 시 클립보드 복사
- 🧠 디버그 모드로 Top 5 예측 결과 + 이미지 표시<br>
디버그 모드 및 일반 모드에서 나타나는 정답은 모두 클릭을 통해 클립보드 복사가 가능합니다.

---

## 🖼️ 실행 예시

### 기본 사용 순서

1. `영역 지정` 버튼 클릭 → 퀴즈 영역 드래그 <br> **영역을 npc, 몬스터 크기에 최대한 맞춰서 설정해주세요**
2. `실행` 버튼 클릭 → 실시간 추론 시작
3. 결과 확인 (클릭 시 정답 복사됨)
4. `중지`, `테스트`, `디버그 모드` 선택적 사용


### 사용 예시
**실행 시 첫 화면**<br>
![image](https://github.com/user-attachments/assets/3532cbf6-a0b4-4488-a01c-d714c41bd3ef)<br><br>
**화면 인식 예시**<br>
영역은 NPC 또는 몬스터의 크기에 최대한 맞춰서 설정하는 것이 정답률 향상에 도움이 됩니다.<br>
![image](https://github.com/user-attachments/assets/853aed7a-d21f-49bb-b9a6-c8bf13ba6f05)<br><br>
**디버그 모드 예시**<br>
디버그 모드는 주로 모델 테스트를 위해 구현된 기능이므로 일반 사용자에게는 필수가 아닙니다.<br>
다만, 기본 모드에서 정답률이 낮다고 느껴질 경우, 디버그 모드를 활성화하면<br>
상위 후보 중 하나에서 정답을 확인할 수 있는 가능성이 있습니다.
![image](https://github.com/user-attachments/assets/a91bb564-c4ef-4a8b-aa9a-9b80e4d2903f)<br><br>
**테스트 모드 예시**<br>
테스트 모드의 경우 기본적으로 설정해둔 테스트 이미지를 통해 간단한 테스트를 할 수 있습니다.
![image](https://github.com/user-attachments/assets/d62f4921-df40-47da-941d-fe2b7fcabc8e)<br>


---

## 🔧 설치 방법

### 1. 의존성 설치

#### 📦 라이브러리 설치
```bash
pip install -r requirements.txt
```

⚡ PyTorch(GPU) 설치<br>
모델 학습 시 GPU를 사용하면 학습 속도가 크게 향상됩니다.<br>
만약 pytorch가 정상적으로 설치되지 않거나 GPU 환경에서 실행하고자 한다면, 아래 명령어로 설치해주세요:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
💡 위 명령어는 CUDA 11.8을 기반으로 하며, 자신의 GPU 환경에 맞는 버전을 확인하는 것이 좋습니다.<br>
공식 가이드: https://pytorch.org/get-started/locally<br><br>


### 2. 🧠 모델 학습

#### 🌱 이미지 다운로드

`util/imgdown.py` 파일을 실행하면 NPC/몬스터 이미지를 자동으로 다운로드하며 `img/` 폴더가 생성됩니다.

```bash
python util/imgdown.py
```

> `answers.json`, `NPC.json`, `mobs.json` 등의 파일이 함께 필요합니다.


#### 🔁 이미지 증강

다운로드된 `img/` 폴더 내 이미지에 대해 증강 및 리사이즈된 이미지를 추가합니다.

```bash
python util/imgbump.py
```

> 각 숫자 폴더별로 증강 이미지가 생성되며, `_resized`, `_bg` 등의 변형 버전이 만들어집니다.


#### 🧠 모델 학습

모델 학습은 `train.py` 파일을 실행하여 진행합니다. ResNet 기반 모델이 학습되며 결과물로 `.pt` 모델 파일이 생성됩니다:

```bash
python train.py
```

> 학습 완료 후 `converted_savedmodel_resnet50/` 폴더 안에 다음 파일이 생성됩니다:
>
> * `model.pt` : 학습된 모델 가중치
> * `labels.txt` : 클래스별 라벨 정보

### 3. 실행

```bash
python speedquiz.py
```
모델을 처음 로딩하는 경우 시간이 걸릴 수 있습니다.

---


## ⚠️ 유의 사항

- 본 프로젝트는 메이플스토리 및 메이플랜드의 **비공식 서포트 도구**입니다.
- 정답 매핑, 이미지 등은 직접 수집한 공개 API 기반 리소스를 사용합니다.
- 학습 데이터 및 모델은 외부 배포하지 않으며, **추론 전용으로 사용**됩니다.

---

## 📜 라이선스

이 프로젝트는 [MIT License](LICENSE)를 따릅니다.

---

## 🙌 기여 및 문의
* 질문, 버그 리포트는 [Issues 탭](https://github.com/Sajandora/Mapleland-SpeedQuiz-AI/issues)에서 남겨주세요.

