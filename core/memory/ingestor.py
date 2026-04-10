import json
import logging
import os
from typing import List, Dict, Any

import aiofiles
import yaml

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
        ext = os.path.splitext(filename)[1].lower()
        logger.info(f"开始摄入文档: {filename}")

        # 1. 读取内容
        content = ""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
        except UnicodeDecodeError:
            # 备选编码读取
            try:
                async with aiofiles.open(file_path, 'r', encoding='gbk', errors='ignore') as f:
                    content = await f.read()
            except Exception as e:
                logger.error(f"文件读取编码错误: {e}", exc_info=True)
                return 0

        if not content.strip():
            return 0

        # 2. 智能切分
        # 特判：如果是结构化数据，使用结构化感知切片；否则使用带重叠项的滑动窗口物理切片
        if ext in ['.json', '.yaml', '.yml']:
            logger.info("检测到结构化文件，启动结构化安全切片策略...")
            raw_chunks = self._chunk_structured_data(content, ext, max_chars=4000)
        else:
            logger.info("启动滑动窗口按行切片策略...")
            raw_chunks = self._chunk_text_physically(content, max_chars=4000, overlap_lines=15)

        logger.info(f"文档已切为 {len(raw_chunks)} 个片段，准备进行 AI 提炼...")

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
        调用 LLM 将文本转化为结构化知识卡片
        """
        system_prompt = f"""
你是一名专业的【知识架构师】。你的任务是将输入的原始文本拆解、提炼为高质量的知识库条目。

当前处理的文件：{filename} (片段 {chunk_index}/{total_chunks})

请严格遵循以下步骤：
1. 分析逻辑：识别文本中的独立知识点、概念定义、操作步骤或核心事实。
2. 绝对保真：如果包含 API 接口、代码片段、JSON 数据结构、URL 或参数说明，请【绝对原样保留】所有大括号、字段名和格式，绝不可精简技术细节！
3. 🚧 截断抛弃机制：由于长文档是分块读取的，你看到的文本头尾可能有被拦腰截断的半句话、不闭合的 JSON 括号或残缺的代码块。对于这些不完整的数据，请**直接无视并抛弃**！相邻的重叠数据块会自动处理完整的版本。绝不要尝试脑补不完整的数据。
4. 独立化：将每个提取出的知识点重写为一段独立的、自包含的文本。（如果是API，必须包含完整路径、方法和核心参数）。
5. 打标：为每个知识点提取 3-5 个关键索引词（Tags）。
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
                {"role": "user", "content": f"请处理以下文本片段：\n\n{text}"}
            ],
            schema=schema
        )
        data = response.get("content", {})
        return data.get("knowledge_units", [])

    def _chunk_text_physically(self, text: str, max_chars: int, overlap_lines: int = 15) -> List[str]:
        """
        物理切片：按行切分，并通过回退 N 行来实现重叠，防止上下文断裂
        """
        chunks = []
        lines = text.split('\n')

        current_chunk_lines = []
        current_len = 0

        i = 0
        while i < len(lines):
            line = lines[i]
            line_len = len(line) + 1  # 算上换行符

            # 如果加上这行超出了限制，且当前块已经有内容了，就打断
            if current_len + line_len > max_chars and current_chunk_lines:
                chunks.append("\n".join(current_chunk_lines))

                # 回退指针，保留 overlap_lines 作为下一个块的开头
                overlap_count = min(overlap_lines, len(current_chunk_lines))
                if overlap_count > 0:
                    current_chunk_lines = current_chunk_lines[-overlap_count:]
                    current_len = sum(len(l) + 1 for l in current_chunk_lines)
                else:
                    current_chunk_lines = []
                    current_len = 0

            current_chunk_lines.append(line)
            current_len += line_len
            i += 1

        if current_chunk_lines:
            chunks.append("\n".join(current_chunk_lines))

        return chunks

    def _chunk_structured_data(self, text: str, ext: str, max_chars: int) -> List[str]:
        """
        结构化切片：将 JSON/YAML 反序列化后，按顶层节点重新组合打包，保证括号绝对闭合
        """
        try:
            if ext == '.json':
                data = json.loads(text)
                separator = ",\n"
                prefix, suffix = "[\n", "\n]" if isinstance(data, list) else ("{\n", "\n}")
            else:
                data = yaml.safe_load(text)
                separator = "\n---\n"
                prefix, suffix = "", ""

            chunks = []

            # 无论外层是 List 还是 Dict，提取其顶层项进行打包
            items = []
            if isinstance(data, list):
                for item in data:
                    item_str = json.dumps(item, ensure_ascii=False) if ext == '.json' else yaml.dump(item,
                                                                                                     allow_unicode=True)
                    items.append(item_str)
            elif isinstance(data, dict):
                for k, v in data.items():
                    item_dict = {k: v}
                    item_str = json.dumps(item_dict, ensure_ascii=False) if ext == '.json' else yaml.dump(item_dict,
                                                                                                          allow_unicode=True)
                    items.append(item_str[1:-1].strip() if ext == '.json' else item_str)  # JSON 剥离外层大括号
            else:
                return [text]  # 无法结构化拆分的标量

            current_chunk = []
            current_len = 0

            for item_str in items:
                if current_len + len(item_str) > max_chars and current_chunk:
                    chunks.append(prefix + separator.join(current_chunk) + suffix)
                    current_chunk = []
                    current_len = 0

                current_chunk.append(item_str)
                current_len += len(item_str)

            if current_chunk:
                chunks.append(prefix + separator.join(current_chunk) + suffix)

            return chunks

        except Exception as e:
            logger.warning(f"结构化解析失败，降级为滑动窗口切片: {e}")
            return self._chunk_text_physically(text, max_chars=max_chars, overlap_lines=15)
