import pandas as pd
import os

# 1. 방금 다운로드 받은 파일 이름
file_path = "train.csv"

print("🔍 VinDr-CXR 정답지(train.csv) 해체를 시작합니다...")
df = pd.read_csv(file_path)

# 전체 데이터 개수 확인
print(f"📊 원본 데이터: 총 {len(df)}개의 전문의 판독 기록이 있습니다.")

# 2. 'No finding(정상)' 노이즈 제거 및 실제 병변만 필터링
# VinDr 데이터에서 class_id 14는 '병변 없음'을 의미하며, 좌표가 비어있습니다.
df_abnormal = df[df['class_id'] != 14].copy()

# 3. 좌표 데이터 정제 (빈칸 제거 및 소수점 잘라내기)
# AI가 정확한 픽셀을 인식할 수 있도록 좌표(x, y)를 깔끔한 정수로 맞춥니다.
df_abnormal.dropna(subset=['x_min', 'y_min', 'x_max', 'y_max'], inplace=True)
df_abnormal[['x_min', 'y_min', 'x_max', 'y_max']] = df_abnormal[['x_min', 'y_min', 'x_max', 'y_max']].astype(int)

print(f"👉 필터링 결과: 실제 병변 타겟 좌표가 있는 기록은 {len(df_abnormal)}개입니다.")

# 4. 순살 마스터 정답지 저장
output_path = "master_vindr_labels.csv"
df_abnormal.to_csv(output_path, index=False)

print(f"🎉 전처리 완료! AI에게 먹일 마스터 파일이 생성되었습니다: {output_path}")