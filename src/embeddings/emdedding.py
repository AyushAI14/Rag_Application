import numpy as np
from sentence_transformers import SentenceTransformer
from src.logging.logging import logger


class EmbeddingManager:
    """It will handle embedding of text by SentenceTransformer"""
    def __init__(self,model_name:str="all-MiniLM-L6-v2"):
        """
        Args : 
            model_name : Huggingface model name for sentence emedding
        """
        self.model_name=model_name
        self.model=None
        self._load_model()
    
    def _load_model(self):
        """Load the SentenceTransformer model"""
        try:
            print(f'Initalizing the embedding model : {self.model_name}')
            self.model = SentenceTransformer(self.model_name)
            print(f'Model of embedding has sucessfully Intialized, Embedding size is {self.model.get_sentence_embedding_dimension()}')
        except Exception as e:
            print(f"Error while Initialing the model : {self.model_name}")
            raise
    def generate_Embedding(self,texts:list[str])->np.ndarray:
        """
        Generate embedding from the text and take list of text as argument
        
        return a list of numpy array with shape(len(text),embedding_dim)
        """
        if not self.model:
            raise ValueError('No Model Given')
        print(f'Generating Embedding for {len(texts)}')
        embeddings = self.model.encode(texts,show_progress_bar=True)
        print(f'Shapeof Embedding are {embeddings.shape}')
        return embeddings

with open('data/text_file/text_split.txt','r',encoding='utf-8') as f:
    chunks = f.read().split('\n')
#Intializing the Embedding 
embeddingManager = EmbeddingManager()
embedding = embeddingManager.generate_Embedding(chunks)