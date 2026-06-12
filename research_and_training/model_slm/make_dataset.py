import json
import random

def generate_cases(num_cases=100):
    surgeries = ["VATS Right Upper Lobectomy", "VATS Left Lower Lobectomy", "VATS Segmentectomy", "Wedge Resection"]
    dataset = []

    for i in range(num_cases):
        pod = random.randint(1, 5)
        surgery = random.choice(surgeries)
        scenario_type = random.choices(["normal", "bleeding", "air_leak", "infection"], weights=[50, 20, 20, 10])[0]

        if scenario_type == "normal":
            input_text = f"POD {pod}일 차 환자 ({surgery}). 활력징후 BP 120/80, HR {random.randint(70,90)}, BT 36.8. Chest tube 배액량 지난 8시간 50cc (Serous). Air leak 없음. 통증 NRS 3점."
            output_text = "현재 활력징후가 안정적이며, 흉관 배액량이 양호하고 Air leak이 관찰되지 않습니다. 정상적인 회복 과정으로 판단됩니다. 통증 조절(NRS 3)을 위해 처방된 경구 진통제를 유지하고, 조기 보행을 격려하십시오."
        
        elif scenario_type == "bleeding":
            input_text = f"POD {pod}일 차 환자 ({surgery}). 활력징후 BP 90/60, HR {random.randint(110,130)} (Tachycardia), BT 37.1. Chest tube 배액량이 최근 1시간 동안 250cc (Sanguineous) 발생함. 어지러움 호소."
            output_text = "🚨 응급 상황입니다. 활력징후 저하와 시간당 200cc 이상의 혈성 흉관 배액은 활동성 출혈(Active bleeding)을 강력히 시사합니다. 즉시 주치의에게 노티하고, 수액 급속 주입을 시작하십시오. 긴급 CBC 및 수혈/재수술 준비가 필요합니다."
            
        elif scenario_type == "air_leak":
            input_text = f"POD {pod}일 차 환자 ({surgery}). 활력징후 안정적. 배액량 100cc 미만(Serous). 기침 시 지속적인 Air leak (Grade 2) 관찰됨. 호흡곤란 없음."
            output_text = "활력징후와 배액량은 양호하나, 지속되는 Air leak이 있습니다. 호흡곤란이 없으므로 현재의 Chest tube 워터씰(Water-seal) 상태를 유지하며 관찰합니다. 흉부 X-ray를 촬영하여 폐의 팽창 상태를 확인하십시오."
            
        else: # infection
            input_text = f"POD {pod}일 차 환자 ({surgery}). 활력징후 BP 110/70, HR 100, BT {round(random.uniform(38.0, 39.0), 1)} (Fever). Chest tube 배액량 양호. 기침 시 누런 가래 관찰됨."
            output_text = "⚠️ 주의 상황입니다. 수술 후 발열과 화농성 객담이 관찰되어 호흡기계 감염(Pneumonia 등)이 의심됩니다. 혈액 배양(Blood culture) 및 객담 배양(Sputum culture) 검사를 즉시 시행하고, 경험적 항생제 투여를 고려하기 위해 주치의에게 노티하십시오."

        # Llama-3 파인튜닝 표준 포맷(Alpaca format)
        dataset.append({
            "instruction": "당신은 흉부외과 전문의를 보조하는 CDSS AI 어시스턴트입니다. 다음 환자의 상태를 분석하고 적절한 임상적 판단과 지시를 내리세요.",
            "input": input_text,
            "output": output_text
        })

    return dataset

# JSONL 파일로 저장
data = generate_cases(100)
with open("train.jsonl", "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 학습용 데이터 100개가 'train.jsonl' 파일로 성공적으로 생성되었습니다!")