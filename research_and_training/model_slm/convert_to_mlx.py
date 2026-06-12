import json
import os

# MLX가 요구하는 데이터 폴더 생성
os.makedirs("mlx_data", exist_ok=True)

print("데이터 변환 시작...")
with open("train.jsonl", "r", encoding="utf-8") as f_in, open("mlx_data/train.jsonl", "w", encoding="utf-8") as f_out:
    for line in f_in:
        data = json.loads(line)
        # MLX가 이해하는 챗봇 대화 형식(messages)으로 구조 변경
        mlx_format = {
            "messages": [
                {"role": "user", "content": f"{data['instruction']}\n\n[환자 상태]\n{data['input']}"},
                {"role": "assistant", "content": data['output']}
            ]
        }
        f_out.write(json.dumps(mlx_format, ensure_ascii=False) + "\n")

# 검증용 데이터(valid.jsonl)도 구조상 필요하므로 동일하게 복사해둡니다.
import shutil
shutil.copyfile("mlx_data/train.jsonl", "mlx_data/valid.jsonl")

print("✅ MLX용 데이터 변환 완료! ('mlx_data' 폴더를 확인하세요)")