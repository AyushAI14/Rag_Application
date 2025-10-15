from chromadb.config import Settings
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
from typing import List,Dict,Any,Tuple
import uuid
import os
import numpy as np
from src.embeddings.emdedding import EmbeddingManager
from src.loaders.loader import Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorStore:
    def __init__(self,collection_name:str="pdf_documents",persist_directory:str="data/vector_store"):
        """
        Args:
            collection_name: Name of the chromeDB collection
            persist_directory: Directry to persist the vector store
        """
        self.collection_name=collection_name
        self.persist_directory = persist_directory
        self.client=None
        self.collection=None
        self._initalize_store()
    def _initalize_store(self):
        """Initialize the Chromedb client and collection"""
        try:
            # Create persistant chromaDB client
            os.makedirs(self.persist_directory,exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Get or create the collection 
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description":"Pdf Document embeddings for RAG"}
            )
            print(f"Vector store intialized, collection name: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error while intializing the vectorDB {e}")
            raise
    
    def add_documents(self,documents: list[Any],embedding:np.ndarray):
        """
        Add document , embedding to to vector db

        Args:
            documents : List of langchain document
            embedding : corresponding embeddings for the documents  
        """
        if len(documents) != len(embedding):
            print("Length of document and embedding are not same.")
        print(f"{len(documents)} documents are being adding to vectorDB")

        # Preparing Data for ChromoDB
        ids=[]
        metadatas=[]
        documents_text=[]
        embedding_text = []

        # feeding the data in specific data list
        for i, (doc,embedd) in enumerate(zip(documents,embedding)):
            # feeding ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # feeding metadata
            md = dict(doc.metadata)
            md['doc_index'] = i
            md['content_lenght']=len(doc.page_content)
            metadatas.append(md)

            #fedding document text
            documents_text.append(doc.page_content)
            #fedding embedding
            embedding_text.append(embedd.tolist())
        try:
            self.collection.add(
                ids=ids,
                embeddings=embedding_text,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total Document stored in vectorDB is {self.collection.count()}")
        except Exception as e:
            print("Error while feeding the document : {e}")
            raise



# ----embedding 
with open('data/text_file/text_split.txt','r',encoding='utf-8') as f:
    chunks = f.read().split('\n')
#Intializing the Embedding 
embeddingManager = EmbeddingManager()
embedding = embeddingManager.generate_Embedding(chunks)
# ----splits
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=200,  
    add_start_index=True,  
)

l = Loader()
docs = l.directory_loader()
all_splits = text_splitter.split_documents(docs)

vectordb = VectorStore()
vectordb.add_documents(all_splits,embedding)