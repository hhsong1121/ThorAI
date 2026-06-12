import requests
import json

print("🚀 Hybrid CDSS API 서버에 환자 데이터를 전송합니다...")

# 1. API 서버 주소 (로컬)
url = "http://127.0.0.1:8000/api/v2/analyze"

# 2. 서버로 보낼 가상 환자 데이터 (출혈 의심 상황)
payload = {
    "surgery_type": "VATS Right Upper Lobectomy",
    "pod_days": 1,
    "vitals": "BP 85/55, HR 125, BT 37.0",
    "drainage": "최근 1시간 220cc (Sanguineous)",
    "symptoms": "어지러움 호소, 식은땀"
}

# 3. 데이터 전송 (POST 요청)
try:
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ 서버 응답 성공!\n")
        print("--- 📚 [RAG] AI가 참고한 가이드라인 일부 ---")
        print(result["used_guideline"])
        print("\n--- 🩺 [MLX] AI의 최종 종합 진단 ---")
        print(result["ai_diagnosis"])
    else:
        print(f"❌ 서버 에러: {response.status_code}")
        print(response.text)
except requests.exceptions.ConnectionError:
    print("❌ 서버 연결 실패: python server.py가 실행 중인지 확인하세요.")