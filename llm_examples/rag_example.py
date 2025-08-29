from typing import List, Dict
import random
import re

# Simulate an Embedding Model
class MockEmbeddingModel:
    def embed_query(self, text: str) -> List[float]:
        """Generates a mock embedding for a query."""
        print(f"DEBUG: Embedding query: '{text[:50]}...'")
        # In a real scenario, this would call a pre-trained embedding model (e.g., Sentence-BERT, OpenAI Embeddings)
        return [float(ord(c)) / 100 for c in text[:128].ljust(128)][:128] # Simple character-based mock embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates mock embeddings for a list of documents."""
        print(f"DEBUG: Embedding documents...")
        # In a real scenario, this would process each document through the embedding model
        return [[float(ord(c)) / 100 for c in t[:128].ljust(128)][:128] for t in texts]

# Simulate a Vector Database
class MockVectorDatabase:
    def __init__(self):
        self.documents = []  # Stores {"embedding": [...], "text": "..."}
        self.texts = []      # For direct lookup in mock similarity_search

    def add_documents(self, embeddings: List[List[float]], texts: List[str]):
        """Adds documents with their embeddings to the mock database."""
        for emb, text in zip(embeddings, texts):
            self.documents.append({"embedding": emb, "text": text})
            self.texts.append(text)
        print(f"DEBUG: Added {len(embeddings)} documents to vector DB.")

    def _calculate_mock_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculates a simple mock similarity (e.g., dot product) between two mock embeddings."""
        # In real vector DBs, this would be optimized cosine similarity or L2 distance
        if not emb1 or not emb2:
            return 0.0
        return sum(e1 * e2 for e1, e2 in zip(emb1, emb2)) / (sum(e**2 for e in emb1)**0.5 * sum(e**2 for e in emb2)**0.5 + 1e-9)

    def similarity_search(self, query_embedding: List[float], k: int = 3) -> List[str]:
        """Performs a mock similarity search and returns top K relevant texts."""
        print(f"DEBUG: Performing similarity search with k={k}...")
        if not self.documents:
            return []

        # Calculate similarity for all stored documents
        similarities = []
        for doc_entry in self.documents:
            similarity = self._calculate_mock_similarity(query_embedding, doc_entry["embedding"])
            similarities.append((similarity, doc_entry["text"]))

        # Sort by similarity in descending order and return top K texts
        similarities.sort(key=lambda x: x[0], reverse=True)
        retrieved_texts = [text for sim, text in similarities[:k]]
        
        # Add a simple heuristic for demonstration if general match is low
        # This makes the mock more "intelligent" for specific queries
        if "巴黎奥运会" in " ".join(retrieved_texts) or any("巴黎奥运会" in t for t in self.texts):
            if "2024年巴黎奥运会将于2024年7月26日开幕" not in " ".join(retrieved_texts):
                for t in self.texts:
                    if "2024年巴黎奥运会将于2024年7月26日开幕" in t:
                        retrieved_texts.append(t)
                        break
        if "RAG技术" in " ".join(retrieved_texts) or any("RAG技术" in t for t in self.texts):
             if "RAG技术结合了信息检索和LLM的生成能力" not in " ".join(retrieved_texts):
                for t in self.texts:
                    if "RAG技术结合了信息检索和LLM的生成能力" in t:
                        retrieved_texts.append(t)
                        break
        
        return list(dict.fromkeys(retrieved_texts))[:k] # Remove duplicates and trim to k

# Simulate a Large Language Model
class MockLLM:
    def invoke(self, prompt: str) -> str:
        """Generates a mock response based on the prompt."""
        print(f"DEBUG: LLM generating response for prompt (first 100 chars): '{prompt[:100]}...'")
        
        # Simple keyword-based response generation
        if "2024年巴黎奥运会" in prompt and "7月26日" in prompt:
            return "2024年巴黎奥运会将于7月26日开幕。"
        elif "RAG技术" in prompt and "信息检索" in prompt:
            return "RAG技术结合了信息检索和大型语言模型的生成能力，旨在提高回答的准确性和可靠性。"
        elif "人工智能技术" in prompt and "发展" in prompt:
            return "人工智能技术发展迅速，RAG技术可以有效增强LLM的知识能力。"
        elif "上下文没有足够的信息" in prompt:
            return "根据提供的上下文，我无法回答这个问题。"
        else:
            return "无法根据提供的信息生成准确回答。"


class RAGSystem:
    def __init__(self, embedding_model: MockEmbeddingModel, vector_db: MockVectorDatabase, llm: MockLLM):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.llm = llm

    def index_documents(self, documents: List[str]):
        """Indexes a list of text documents into the vector database."""
        # Simple chunking for this example: each document is a chunk
        chunks = documents 
        embeddings = self.embedding_model.embed_documents(chunks)
        self.vector_db.add_documents(embeddings, chunks)
        print("INFO: Documents indexed successfully.")

    def query(self, user_question: str) -> str:
        """Processes a user question using the RAG workflow."""
        print(f"\nINFO: Processing query: '{user_question}'")
        
        # 1. Retrieval Phase
        query_embedding = self.embedding_model.embed_query(user_question)
        retrieved_docs = self.vector_db.similarity_search(query_embedding, k=2)  # Retrieve top 2 relevant documents

        if not retrieved_docs:
            print("WARNING: No relevant documents retrieved. Relying solely on LLM's internal knowledge (may hallucinate).")
            # If no relevant documents are found, still pass the question to the LLM, but note the lack of context.
            context_prompt_part = ""
            instruction = "请直接回答以下问题。请注意，没有提供额外上下文信息。"
        else:
            context = "\n".join(retrieved_docs)
            print(f"INFO: Retrieved context:\n---\n{context}\n---")
            context_prompt_part = f"上下文：\n{context}\n\n"
            instruction = "请根据以下提供的上下文信息，简洁、准确地回答问题。如果上下文没有足够的信息，请说明。"
        
        # 2. Generation Phase: Construct the Augmented Prompt
        full_prompt = (
            f"{instruction}\n\n"
            f"{context_prompt_part}"
            f"问题：{user_question}\n\n"
            f"回答："
        )
        
        # 3. LLM Generates Answer
        final_answer = self.llm.invoke(full_prompt)
        print(f"INFO: LLM generated answer.")
        return final_answer

# Main execution block
if __name__ == "__main__":
    # Initialize RAG System Components
    embedding_model = MockEmbeddingModel()
    vector_db = MockVectorDatabase()
    llm = MockLLM()
    rag_system = RAGSystem(embedding_model, vector_db, llm)

    # Prepare Knowledge Base Documents
    knowledge_base_docs = [
        "2024年巴黎奥运会将于2024年7月26日开幕，闭幕式在8月11日。本届奥运会是历史上首次将开幕式放在体育场外举行的奥运会。",
        "巴黎奥运会是第33届夏季奥林匹克运动会，在法国巴黎举行。帆船和冲浪比赛将在马赛和塔希提岛进行。",
        "RAG技术结合了信息检索和LLM的生成能力，旨在提高回答的准确性和可靠性。",
        "LLM可能存在知识截止日期和幻觉问题，RAG是解决这些问题的有效方案。",
        "向量数据库是RAG系统的核心组件之一，用于高效存储和检索文档嵌入。",
        "人工智能（AI）领域正在快速发展，其中大型语言模型和RAG是当前的热点研究方向。",
        "地球引力是物体之间相互吸引的力，大小与物体质量成正比，与它们之间距离的平方成反比。这是艾萨克·牛顿爵士提出的万有引力定律。"
    ]

    # Index Documents
    rag_system.index_documents(knowledge_base_docs)

    print("\n" + "="*80 + "\n")

    # Run Queries
    question1 = "2024年巴黎奥运会开幕日期是什么时候？"
    answer1 = rag_system.query(question1)
    print(f"\n问题: {question1}\n回答: {answer1}")

    print("\n" + "="*80 + "\n")

    question2 = "RAG技术的核心优势是什么？"
    answer2 = rag_system.query(question2)
    print(f"\n问题: {question2}\n回答: {answer2}")

    print("\n" + "="*80 + "\n")

    question3 = "关于地球重力的理论是什么？"
    answer3 = rag_system.query(question3)
    print(f"\n问题: {question3}\n回答: {answer3}")
    
    print("\n" + "="*80 + "\n")

    question4 = "什么是量子力学？" # Knowledge base does not have this specific info
    answer4 = rag_system.query(question4)
    print(f"\n问题: {question4}\n回答: {answer4}")
    
    print("\n" + "="*80 + "\n")
