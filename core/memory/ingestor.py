import json
import logging
import os
from typing import List, Dict, Any

from core.infrastructure.api_client import GenericAPIClient
from core.memory.schema import SemanticMemory
from core.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


class KnowledgeIngestor:
    def __init__(self, api_client: GenericAPIClient, vector_store: VectorStore):
        self.api_client = api_client
        self.vector_store = vector_store

    async def ingest_file(self, file_path: str, user_id: str = "system") -> int:
        """
        读取文件，通过 LLM 智能提炼、打标、分块，存入知识库。
        返回生成的知识条目数量。
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")

        filename = os.path.basename(file_path)
        logger.info(f"开始摄入文档: {filename}")

        # 1. 读取内容
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
                content = f.read()

        if not content.strip():
            return 0

        # 2. 粗切分 (Coarse Chunking)
        # 为了防止超过 LLM 上下文限制，先按字符数进行物理切分
        # 建议 3000-5000 字符，留足 token 给 LLM 思考和输出 JSON
        raw_chunks = self._chunk_text_physically(content, max_chars=4000)
        logger.info(f"文档已粗切为 {len(raw_chunks)} 个片段，准备进行 AI 提炼...")

        total_units = 0

        # 3. LLM 智能处理循环
        for i, raw_text in enumerate(raw_chunks):
            try:
                # 调用 LLM 进行精细化处理
                knowledge_units = await self._refine_chunk_with_llm(raw_text, filename, i + 1, len(raw_chunks))

                # 4. 存入向量数据库
                for unit in knowledge_units:
                    refined_content = unit.get("content")
                    keywords = unit.get("keywords", [])

                    if not refined_content:
                        continue

                    # 加上来源标注，方便溯源
                    final_content = f"【知识库: {filename}】\n{refined_content}"

                    # 创建语义记忆对象
                    memory = SemanticMemory(
                        content=final_content,
                        keywords=keywords
                    )

                    # 存入
                    await self.vector_store.save_vector_memory(memory, user_id)
                    total_units += 1

                logger.info(f"片段 {i + 1}/{len(raw_chunks)} 处理完成，生成 {len(knowledge_units)} 条知识。")

            except Exception as e:
                logger.error(f"处理片段 {i + 1} 时出错: {e}", exc_info=True)

        logger.info(f"文档 {filename} 摄入完成，共生成 {total_units} 条知识条目。")
        return total_units

    async def _refine_chunk_with_llm(self, text: str, filename: str, chunk_index: int, total_chunks: int) -> List[
        Dict[str, Any]]:
        """
        调用 LLM 将粗糙文本转化为结构化知识卡片
        """
        system_prompt = f"""
你是一名专业的【知识架构师】。你的任务是将输入的原始文本拆解、提炼为高质量的知识库条目。

当前处理的文件：{filename} (片段 {chunk_index}/{total_chunks})

请遵循以下步骤：
1. 分析逻辑：识别文本中的独立知识点、概念定义、操作步骤或核心事实。
2. 去除噪音：删除无意义的格式字符、口语废话、页眉页脚等。
3. 独立化：将每个知识点重写为一段独立的、自包含的文本。即使脱离上下文，这段话也应该是通顺且信息完整的。
4. 打标：为每个知识点提取 3-5 个关键索引词（Tags）。

输出必须严格遵守 JSON 格式：
{{
  "knowledge_units": [
    {{
      "content": "这里是提炼后的知识内容，包含事实、定义或描述。",
      "keywords": ["关键词1", "关键词2", "关键词3"]
    }},
    ...
  ]
}}
"""
        # 使用 JSON Schema 强制约束输出
        schema = {
            "type": "object",
            "properties": {
                "knowledge_units": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "Refined knowledge content"},
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Tags for indexing"
                            }
                        },
                        "required": ["content", "keywords"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["knowledge_units"],
            "additionalProperties": False
        }

        response = await self.api_client.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请处理以下文本：\n\n{text}"}
            ],
            schema=schema
        )

        try:
            # 解析 JSON
            content_str = response["choices"][0]["message"]["content"]
            data = json.loads(content_str)
            return data.get("knowledge_units", [])
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"LLM 返回的 JSON 格式错误: {e}")
            return []

    def _chunk_text_physically(self, text: str, max_chars: int) -> List[str]:
        """
        物理切片：按换行符优先切分，防止截断句子
        """
        chunks = []
        current_chunk = ""

        for paragraph in text.split('\n'):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(current_chunk) + len(paragraph) < max_chars:
                current_chunk += paragraph + "\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph + "\n"

        if current_chunk:
            chunks.append(current_chunk)

        return chunks
