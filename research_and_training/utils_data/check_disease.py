import pandas as pd

# 마스터 정답지 불러오기
df = pd.read_csv("master_vindr_labels.csv")

print("\n🏥 [VinDr-CXR] 흉부 병변 데이터 분포 브리핑")
print("=" * 45)

# 병변 이름(class_name)별로 개수를 세어서 내림차순 정렬
disease_counts = df['class_name'].value_counts()

# 결과 예쁘게 출력
for disease, count in disease_counts.items():
    print(f"🩺 {disease:<25} : {count:>5} 건")

print("=" * 45)
print("💡 팁: 데이터가 너무 적은 희귀 병변은 나중에 AI 모델링 시")
print("   '데이터 증강(Augmentation)' 기술을 적용해야 합니다.\n")