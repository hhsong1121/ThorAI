from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

print("🚀 엔진 및 DB 연결 중... (문서를 다시 읽지 않아 매우 빠릅니다!)")

# 1. 모델 세팅 (이번엔 Temperature를 낮춰서 헛소리를 줄입니다)
llm = Ollama(model="llama3", request_timeout=120.0, temperature=0.1)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
Settings.llm = llm
Settings.embed_model = embed_model

# 2. 어제 만들어둔 Qdrant DB 불러오기
client = qdrant_client.QdrantClient(path="./qdrant_db")
vector_store = QdrantVectorStore(client=client, collection_name="thoracic_guidelines")

# 문서를 다시 파싱하지 않고, 기존 DB에서 인덱스만 쏙 빼옵니다.
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# 3. ⭐️ 프롬프트 엔지니어링 (한국어 강제 및 역할 부여)
qa_prompt_tmpl_str = (
    "너는 흉부외과 전문의를 보조하는 AI 어시스턴트야.\n"
    "아래 제공된 [참고 문서] 정보만을 사용하여 질문에 답해.\n"
    "문서에 없는 내용을 지어내지 마.\n"
    "대답은 반드시 '한국어(Korean)'로, 의학 전문가처럼 명확하고 간결하게 작성해.\n\n"
    "[참고 문서]\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "질문: {query_str}\n"
    "대답: "
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

# 4. 쿼리 엔진 세팅 및 질문 던지기
query_engine = index.as_query_engine(text_qa_template=qa_prompt_tmpl)

question = "이 문서에서 설명하는 ERAS(Enhanced Recovery After Surgery) 또는 수술 후 관리에 대한 핵심 내용을 요약해줘."
print(f"\n👨‍⚕️ 질문: {question}")
print("🤖 AI가 답변을 고민 중입니다...\n")

response = query_engine.query(question)
print("--- 🩺 AI의 답변 ---")
print(response)