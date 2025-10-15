from typing import List,Any
from src.vector_store.vector_store import VectorStore
from src.embeddings.emdedding import EmbeddingManager

class RetreivalManager:
    """
    This class will Retrieval similar context with query from VectorStore
    """
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        print('Retreival has been Intialized')
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    def retreive(self,query:str,top_k:int=5,score_threshold:float=0.0)->List[dict[str,Any]]:
        """
        Retreive the data 

        Arg:
            query:given by the user
            top_k:top k similar context
            score_threshold : minimum similarity score
         Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")
        embedded_query = self.embedding_manager.generate_Embedding([query])[0]
        try:
            result = self.vector_store.collection.query(
                query_embeddings=[embedded_query.tolist()],
                n_results=top_k
            )
            retreived_docs = []
            if result['documents'] and result['documents'][0]:
                ids = result['ids'][0]
                distances = result['distances'][0]
                documents = result['documents'][0]
                metadatas = result['metadatas'][0]
                for i, (doc_id, distance, document, metadata) in enumerate(zip(ids, distances, documents, metadatas)):
                    similarity_score = 1-distance  # distance (0 = same, 1 = far) → into similarity score (1 = same, 0 = far).
                    if similarity_score>=score_threshold:
                        retreived_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                print(f'{len(retreived_docs)} retreived from vectordb')
                return retreived_docs
            else:
                print('No Document found')
        except Exception as e:
            print(f'Error while retreiving the data : {e}')
            raise

vectordb = VectorStore()
embeddingManager = EmbeddingManager()
rm = RetreivalManager(vectordb, embeddingManager)
# print(rm.retreive(""))