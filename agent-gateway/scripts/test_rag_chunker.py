from app.rag.chunker import TextChunker


def main() -> None:
    print("========== RAG CHUNKER TEST ==========")

    # 1. 短文本
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    result = chunker.split("这是一段短文本。")

    assert len(result) == 1
    print("[PASS] 短文本不拆分")

    # 2. 长文本
    text = "这是用于测试文本切片的内容。" * 100

    chunks = chunker.split(text)

    assert len(chunks) > 1
    assert all(chunks)
    assert all(len(chunk) <= 100 for chunk in chunks)

    print("[PASS] 长文本成功拆分")
    print(f"[INFO] Chunk 数量: {len(chunks)}")

    # 3. 重叠检查
    overlap_chunker = TextChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    overlap_chunks = overlap_chunker.split("A" * 250)

    assert len(overlap_chunks) > 1
    assert overlap_chunks[0][-20:] == overlap_chunks[1][:20]

    print("[PASS] Chunk 重叠正常")

    # 4. 空文本
    assert chunker.split("") == []
    assert chunker.split("   ") == []

    print("[PASS] 空文本处理正常")

    # 5. 非法配置
    try:
        TextChunker(chunk_size=100, chunk_overlap=100)
        raise AssertionError("非法配置未被拦截")
    except ValueError:
        print("[PASS] 非法切片参数拦截")

    print("=====================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    main()