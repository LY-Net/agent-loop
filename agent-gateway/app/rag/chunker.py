"""RAG document processing and retrieval."""
import re
class TextChunker:
    """基于字符长度的文本切片器。"""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        """
        将长文本拆分为多个 Chunk。

        优先在段落、句子和空白位置切分。
        如果找不到合适边界，则按字符长度切分。
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        text = text.strip()

        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            hard_end = min(start + self.chunk_size, text_length)
            end = hard_end

            if hard_end < text_length:
                boundary = self._find_boundary(
                    text=text,
                    start=start,
                    end=hard_end,
                )

                if boundary > start:
                    end = boundary

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            next_start = end - self.chunk_overlap

            # 防止因边界选择导致没有前进。
            if next_start <= start:
                next_start = end

            start = next_start

        return chunks

    @staticmethod
    def _find_boundary(
        text: str,
        start: int,
        end: int,
    ) -> int:
        """
        在当前窗口后半段寻找优先切分边界。

        优先级：
        1. 段落
        2. 中文或英文句子结束符
        3. 换行
        4. 空格
        """

        window_start = start + (end - start) // 2
        window = text[window_start:end]

        patterns = [
            r"\n\s*\n",
            r"[。！？!?；;]\s*",
            r"\n",
            r"\s+",
        ]

        for pattern in patterns:
            matches = list(re.finditer(pattern, window))

            if matches:
                match = matches[-1]
                return window_start + match.end()

        return end