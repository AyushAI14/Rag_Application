from src.logging.logging import logger
from src.vector_store.vector_store import VectorStore
from src.embeddings.emdedding import EmbeddingManager
from src.retrieval.retreive import RetreivalManager
from src.llm.generation import GeminiLLM

#Abhi file loader bhi lana hai

logger.info("-------- Initalizing Vector Store ------")
vectordb = VectorStore()
logger.info("------ Initalizing Embedding Manager--------")
embeddingManager = EmbeddingManager()
logger.info("-------- Retreving the Context from Vector Store by Embedding Query ---------")
rm = RetreivalManager(vectordb, embeddingManager)
r = GeminiLLM()
print(r.rag_simple("Explain Expansion",rm))