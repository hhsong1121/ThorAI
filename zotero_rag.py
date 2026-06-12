import os
import re
from pyzotero import zotero
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class GuidelineRAG:
    def __init__(self, library_id, api_key, target_tag="CDSS"):
        # API 연결 오류를 방지하기 위해 ID 공백 제거
        clean_id = str(library_id).strip()
        clean_key = str(api_key).strip()
        
        self.zot = zotero.Zotero(clean_id, 'user', clean_key)
        self.target_tag = target_tag
        self.vector_store = None
        
        print("🧠 의학 전문 임베딩 모델(PubMedBERT)을 로딩 중입니다...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="NeuML/pubmedbert-base-embeddings" 
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )

    def sync_and_build_knowledge_base(self):
        print("📚 Zotero에서 가이드라인을 동기화하고 스마트 청킹을 시작합니다...")
        
        try:
            items = self.zot.items(tag=self.target_tag)
        except Exception as e:
            print(f"🚨 Zotero API 연결 실패! 에러: {e}")
            return

        documents = []
        
        for item in items:
            item_type = item['data'].get('itemType', '')
            
            # 🚨 [수정됨] 'note' 뿐만 아니라 'annotation(형광펜)' 데이터도 수집합니다!
            if item_type in ['note', 'annotation']:
                self._process_text_item(item, documents)
            
            elif item_type != 'attachment': 
                try:
                    children = self.zot.children(item['data']['key'])
                    for child in children:
                        child_type = child['data'].get('itemType')
                        # 🚨 [수정됨] 자식 항목 중에서도 형광펜 데이터를 수집합니다.
                        if child_type in ['note', 'annotation']:
                            self._process_text_item(child, documents, parent_item=item)
                except Exception as e:
                    pass
        
        if documents:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            print(f"✅ 성공! 총 {len(documents)}개의 의학 문맥 조각(Chunks)이 벡터 DB에 저장되었습니다.")
        else:
            print("⚠️ 텍스트를 찾을 수 없습니다. Zotero에서 형광펜(Highlight)이나 노트를 확인해주세요.")

    # 🚨 [수정됨] 기존 _process_note 함수를 형광펜까지 처리할 수 있게 이름과 기능을 변경했습니다.
    def _process_text_item(self, text_item, documents, parent_item=None):
        raw_text = ""
        item_type = text_item['data'].get('itemType', '')
        item_title = text_item['data'].get('title', '제목없음')
        
        # 1. 원본 텍스트 추출 및 확인
        if item_type == 'note':
            raw_text = text_item['data'].get('note', '')
            print(f"    📝 [노트 분석 중] 길이: {len(raw_text)}자 발견")
        elif item_type == 'annotation':
            highlighted = text_item['data'].get('annotationText', '')
            comment = text_item['data'].get('annotationComment', '')
            raw_text = f"{highlighted} {comment}".strip()
            print(f"    🖍️ [형광펜 분석 중] 길이: {len(raw_text)}자 발견")

        if not raw_text.strip():
            print("    ❌ 텍스트가 비어있어 무시됩니다.")
            return

        # 🚨 [핵심 해결] 정규식(Regex)을 이용해 모든 HTML 태그를 공백으로 치환합니다.
        # <...> 형태로 된 모든 코드를 날려버리고 순수 글자만 남깁니다.
        clean_text = re.sub(r'<[^>]+>', ' ', raw_text) 
        clean_text = clean_text.replace('&nbsp;', ' ').strip()
        
        print(f"    ✨ [정제 완료] 추출된 내용: {clean_text[:50]}...")
        
        # 메타데이터 (출처) 달기
        parent_title = "Unknown Guideline"
        if parent_item:
            parent_title = parent_item['data'].get('title', 'Unknown Guideline')
        elif 'parentItem' in text_item['data']:
            try:
                parent = self.zot.item(text_item['data']['parentItem'])
                parent_title = parent['data'].get('title', 'Unknown Guideline')
            except:
                pass
                
        metadata = {
            "source": parent_title,
            "zotero_key": text_item['data']['key']
        }
        
        # 텍스트 청킹(자르기) 및 저장
        chunks = self.text_splitter.split_text(clean_text)
        for chunk in chunks:
            if chunk.strip():
                documents.append(Document(page_content=chunk, metadata=metadata))

    def get_clinical_evidence(self, query: str, top_k: int = 3) -> str:
        if not self.vector_store:
            return "가이드라인 지식베이스가 로드되지 않았습니다."
            
        docs = self.vector_store.similarity_search(query, k=top_k)
        
        if not docs:
            return "환자 상태에 부합하는 명확한 가이드라인을 찾지 못했습니다."
            
        evidence_text = "**[관련 임상 가이드라인 및 근거]**\n\n"
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "출처 미상")
            content = doc.page_content.strip().replace('\n', ' ')
            evidence_text += f"> {i+1}. {content} *(출처: {source})*\n\n"
            
        return evidence_text