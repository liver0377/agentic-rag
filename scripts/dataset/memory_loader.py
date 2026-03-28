"""Memory 预置数据加载器。

将 preset_memories.json 中的记忆数据导入到 ChromaDB 向量数据库。

Usage:
    python scripts/dataset/memory_loader.py --input preset_memories.json
    python scripts/dataset/memory_loader.py --input preset_memories.json --clear-existing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MemoryLoader:
    """预置记忆数据加载器。

    负责将 JSON 格式的记忆数据导入到 ChromaDB 向量数据库。
    """

    def __init__(
        self,
        persist_directory: str = "./data/memory_test_db",
        collection_name: str = "memory_test",
        embedding_provider: str = "mock",
    ):
        """初始化加载器。

        Args:
            persist_directory: ChromaDB 持久化目录
            collection_name: 集合名称
            embedding_provider: Embedding 提供商 ("mock" | "real")
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider
        self._client = None
        self._collection = None
        self._embedding_func = None

    def _init_chroma(self) -> None:
        """初始化 ChromaDB 客户端和集合。"""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as e:
            raise ImportError("请安装 chromadb: pip install chromadb") from e

        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(f"ChromaDB 初始化完成: {self.persist_directory}")

    def _get_embedding_func(self):
        """获取 Embedding 函数。"""
        if self._embedding_func is not None:
            return self._embedding_func

        if self.embedding_provider == "mock":
            self._embedding_func = self._mock_embedding
        else:
            self._embedding_func = self._real_embedding

        return self._embedding_func

    def _mock_embedding(self, texts: List[str]) -> List[List[float]]:
        """Mock Embedding 函数，用于测试。

        基于文本内容生成确定性向量，保证相同文本生成相同向量。
        """
        import hashlib

        vectors = []
        for text in texts:
            hash_obj = hashlib.sha256(text.encode("utf-8"))
            vector = [
                float(int(hash_obj.hexdigest()[i : i + 2], 16)) / 255.0 for i in range(0, 64, 2)
            ]
            vector = vector[:384]
            while len(vector) < 384:
                vector.append(0.0)
            vectors.append(vector)
        return vectors

    def _real_embedding(self, texts: List[str]) -> List[List[float]]:
        """真实 Embedding 函数，使用配置的 Embedding 模型。"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("sentence-transformers 未安装，使用 mock embedding")
            return self._mock_embedding(texts)

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def load_preset_memories(self, filepath: str) -> Dict[str, Any]:
        """加载预置记忆 JSON 文件。

        Args:
            filepath: JSON 文件路径

        Returns:
            解析后的数据字典
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"加载预置记忆: {filepath}")
        return data

    def clear_existing(self) -> int:
        """清除现有记忆数据。

        Returns:
            删除的记录数
        """
        if self._collection is None:
            self._init_chroma()

        count = self._collection.count()
        if count > 0:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"已清除 {count} 条现有记忆")

        return count

    def import_memories(
        self,
        data: Dict[str, Any],
        clear_existing: bool = False,
    ) -> Dict[str, int]:
        """导入记忆数据到 ChromaDB。

        Args:
            data: 预置记忆数据
            clear_existing: 是否清除现有数据

        Returns:
            导入统计信息
        """
        if self._collection is None:
            self._init_chroma()

        if clear_existing:
            self.clear_existing()

        stats = {
            "total_memories": 0,
            "users": 0,
            "by_type": {},
        }

        all_records = []
        all_embeddings = []
        all_metadatas = []
        all_documents = []

        users = data.get("users", [])
        embed_func = self._get_embedding_func()

        for user_data in users:
            user_id = user_data["user_id"]
            memories = user_data.get("memories", [])
            stats["users"] += 1

            for memory in memories:
                memory_id = memory["id"]
                memory_type = memory["type"]
                content = memory["content"]
                importance = memory.get("importance", 1.0)
                metadata = memory.get("metadata", {})
                role = memory.get("role")

                full_metadata = {
                    "user_id": user_id,
                    "type": memory_type,
                    "content": content,
                    "importance": importance,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "role": role or "",
                    **{
                        k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                        for k, v in metadata.items()
                    },
                }

                all_records.append(memory_id)
                all_metadatas.append(full_metadata)
                all_documents.append(content)

                stats["by_type"][memory_type] = stats["by_type"].get(memory_type, 0) + 1
                stats["total_memories"] += 1

        if all_documents:
            logger.info(f"生成 Embedding: {len(all_documents)} 条记忆...")
            all_embeddings = embed_func(all_documents)

            self._collection.upsert(
                ids=all_records,
                embeddings=all_embeddings,
                metadatas=all_metadatas,
                documents=all_documents,
            )

            logger.info(f"成功导入 {stats['total_memories']} 条记忆到 ChromaDB")

        return stats

    def query_memories(
        self,
        query_text: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """查询记忆。

        Args:
            query_text: 查询文本
            user_id: 用户 ID 过滤
            top_k: 返回数量

        Returns:
            匹配的记忆列表
        """
        if self._collection is None:
            self._init_chroma()

        embed_func = self._get_embedding_func()
        query_embedding = embed_func([query_text])[0]

        where_filter = None
        if user_id:
            where_filter = {"user_id": user_id}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "distances", "documents"],
        )

        memories = []
        if results and results["ids"] and results["ids"][0]:
            for i, memory_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results.get("distances") else 0.0
                score = 1.0 - (distance / 2.0)
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                document = results["documents"][0][i] if results.get("documents") else ""

                memories.append(
                    {
                        "id": memory_id,
                        "score": max(0.0, score),
                        "content": document,
                        "metadata": metadata,
                    }
                )

        return memories

    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息。"""
        if self._collection is None:
            self._init_chroma()

        return {
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
            "total_records": self._collection.count(),
        }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Memory 预置数据加载器")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="scripts/dataset/preset_memories.json",
        help="输入 JSON 文件路径",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./data/memory_test_db",
        help="ChromaDB 持久化目录",
    )
    parser.add_argument(
        "--collection",
        "-c",
        type=str,
        default="memory_test",
        help="集合名称",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="导入前清除现有数据",
    )
    parser.add_argument(
        "--embedding",
        "-e",
        type=str,
        choices=["mock", "real"],
        default="mock",
        help="Embedding 提供商 (mock 用于快速测试)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="仅显示统计信息",
    )

    args = parser.parse_args()

    input_path = (
        PROJECT_ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    )

    loader = MemoryLoader(
        persist_directory=args.output_dir,
        collection_name=args.collection,
        embedding_provider=args.embedding,
    )

    if args.stats:
        stats = loader.get_stats()
        print(f"\n📊 Memory 数据库统计:")
        print(f"   集合名称: {stats['collection_name']}")
        print(f"   持久化目录: {stats['persist_directory']}")
        print(f"   总记录数: {stats['total_records']}")
        return 0

    try:
        data = loader.load_preset_memories(str(input_path))
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        return 1

    stats = loader.import_memories(data, clear_existing=args.clear_existing)

    print(f"\n✅ 导入完成:")
    print(f"   总记忆数: {stats['total_memories']}")
    print(f"   用户数: {stats['users']}")
    print(f"   按类型分布:")
    for mem_type, count in stats.get("by_type", {}).items():
        print(f"     - {mem_type}: {count}")

    print(f"\n📍 持久化目录: {args.output_dir}")
    print(f"📍 集合名称: {args.collection}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
