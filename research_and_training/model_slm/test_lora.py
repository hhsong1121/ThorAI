from mlx_lm import load, generate

# 1. 베이스 모델과 방금 학습한 '흉부외과 어댑터'를 같이 불러옵니다.
model_path = "mlx-community/Qwen2.5-7B-Instruct-4bit"
adapter_path = "./adapters"

print("🧠 베이스 모델에 흉부외과 지식(Adapter)을 장착하는 중...")
model, tokenizer = load(model_path, adapter_path=adapter_path)

# 2. 모델에게 던질 새로운 테스트 케이스 (학습 데이터에 없던 미세하게 다른 수치)
instruction = "당신은 흉부외과 전문의를 보조하는 CDSS AI 어시스턴트입니다. 다음 환자의 상태를 분석하고 적절한 임상적 판단과 지시를 내리세요."
patient_status = "POD 1일 차 환자 (VATS Right Upper Lobectomy). 활력징후 BP 85/55, HR 125, BT 37.0. Chest tube 배액량 최근 1시간 220cc (Sanguineous). 식은땀을 흘리며 어지러움을 호소함."

# 3. 모델이 이해할 수 있는 채팅 형식으로 변환
messages = [{"role": "user", "content": f"{instruction}\n\n[환자 상태]\n{patient_status}"}]
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("\n👨‍⚕️ [테스트 상황 입력]")
print(patient_status)
print("\n🤖 [AI 진단 생성 중...]")

# 4. 답변 생성 (max_tokens로 답변 길이 제한)
response = generate(model, tokenizer, prompt=prompt_text, max_tokens=300, verbose=False)

print("\n--- 🩺 AI의 진단 및 지시 ---")
print(response)