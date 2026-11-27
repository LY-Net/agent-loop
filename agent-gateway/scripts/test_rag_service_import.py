from app.rag.chunker import TextChunker
from app.rag.repository import RagRepository
from app.rag.service import RagService
from app.rag.schemas import (
    RagDocumentIngestRequest,
    RagDocumentIngestResponse,
    RagDocumentDeleteResponse,
)

print("========== RAG MODULE IMPORT TEST ==========")

assert TextChunker is not None
assert RagRepository is not None
assert RagService is not None
assert RagDocumentIngestRequest is not None
assert RagDocumentIngestResponse is not None
assert RagDocumentDeleteResponse is not None

print("[PASS] RAG Chunker 导入成功")
print("[PASS] RAG Repository 导入成功")
print("[PASS] RAG Service 导入成功")
print("[PASS] RAG Schemas 导入成功")

print("============================================")
print("Result: ALL PASSED")