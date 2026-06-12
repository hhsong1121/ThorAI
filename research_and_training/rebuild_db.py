from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import qdrant_client
import os
import shutil

print("🧹 1. 기존의 오염된 DB를 청소합니다...")
if os.path.exists("../database/qdrant_db"):
    shutil.rmtree("../database/qdrant_db")

print("📄 2. PyMuPDF 엔진으로 가이드라인(PDF)의 순수 텍스트만 정밀 추출 중...")
loader = PyMuPDFReader()

try:
    documents = SimpleDirectoryReader(
        "./data", 
        file_extractor={".pdf": loader}
    ).load_data()
except ValueError as e:
    print(f"🚨 에러: ./data 폴더를 찾을 수 없거나 파일이 없습니다. ({e})")
    exit()

print(f"👉 [체크포인트] 총 {len(documents)}개의 페이지/문서를 성공적으로 읽었습니다!")

if len(documents) == 0:
    print("🚨 에러: 읽을 수 있는 데이터가 0개입니다! data 폴더 안에 PDF가 있는지 확인하세요.")
    exit()

print("🧠 3. 임베딩 모델 로딩 중...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
Settings.llm = None 

print("💾 4. 깨끗한 데이터로 Qdrant DB를 다시 구축합니다...")
client = qdrant_client.QdrantClient(path="../database/qdrant_db")
vector_store = QdrantVectorStore(client=client, collection_name="thoracic_guidelines")

# 🚨 제가 빼먹었던 핵심 코드입니다 (StorageContext 다리 연결)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context # 이제 허공이 아니라 실제 하드디스크 DB로 들어갑니다!
)

print("✅ 깨끗한 RAG DB 재생성 완료! 이제 서버를 켜셔도 좋습니다.")