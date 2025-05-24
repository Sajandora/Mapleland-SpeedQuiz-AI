import json
import requests
import os

# 파일 경로
ANS_PATH = './answers.json'
NPC_PATH = './NPC.json'
MOB_PATH = './mobs.json'
BASE_SAVE_DIR = './img'

# JSON 파일 읽기
with open(ANS_PATH, 'r', encoding='utf-8') as f:
    ans_list = json.load(f)

with open(NPC_PATH, 'r', encoding='utf-8') as f:
    npc_list = json.load(f)

with open(MOB_PATH, 'r', encoding='utf-8') as f:
    mob_list = json.load(f)

# 이름 -> 코드 매핑
npc_name_to_code = {entry['name_ko'].replace(' ', ''): entry['code'] for entry in npc_list}
mob_name_to_code = {entry['name_ko'].replace(' ', ''): entry['code'] for entry in mob_list}

# 다운로드 시작
for entry in ans_list:
    img_id = str(entry['img'])
    answer_name = entry['answer'].replace(' ', '')

    # NPC 또는 MOB 선택
    if int(img_id) <= 298:
        code = npc_name_to_code.get(answer_name)
        url_template = "https://maplestory.io/api/gms/62/npc/{code}/icon"
    else:
        code = mob_name_to_code.get(answer_name)
        url_template = "https://maplestory.io/api/gms/62/mob/{code}/icon"

    if not code:
        print(f"[X] 코드 매칭 실패: {img_id} ({answer_name})")
        continue

    # 저장 경로 생성
    target_dir = os.path.join(BASE_SAVE_DIR, img_id)
    os.makedirs(target_dir, exist_ok=True)
    save_path = os.path.join(target_dir, f"{img_id}.png")

    url = url_template.format(code=code)

    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"[O] 다운로드 성공: {img_id} ({answer_name})")
        else:
            print(f"[X] 다운로드 실패: {img_id} ({answer_name}) - 상태코드 {response.status_code}")
    except Exception as e:
        print(f"[X] 요청 오류: {img_id} ({answer_name}) - {e}")

print("\n✅ 모든 이미지 다운로드 완료!")
