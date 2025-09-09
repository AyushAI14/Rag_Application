# What is RAG Applications?

A way to optimize LLMs so they gain experience from an organization’s internal data, beyond their training data.

---

# Why RAG?

- Extends the scope of an LLM on a specific domain  
- Cost-efficient  
- More accurate  

---

# Solutions that RAG Provides

### Problem 1
Taking a recent model **GPT-5** which was launched in August — it is trained only on data up to that date.  
If we try to ask a question related to **September**, it may start hallucinating (making things up).  

### Problem 2
Consider a startup that has **policies, finance, HR data** as confidential information and wants to build a chatbot.  

- Fine-tuning a model would be **expensive and time-consuming**  
- Data would keep **updating continuously**  

---

### So, here comes RAG

- Accurate  
- Updates every time  
- Merges with LLM to provide better results  

---

# Working of RAG

<img width="1280" height="853" alt="RAG Workflow" src="https://github.com/user-attachments/assets/eb1b3a52-a804-4f6b-9055-7610244d272b" />


# Implementation of RAG
## Process 1 : Indexing
#### Implementation Steps
1. **Data Collection**: Gather relevant data from the organization's internal systems.
2. **Data Splitting**: Converting data into chunks.
2. **Embedding Generation**: Generate embeddings for the collected data using a suitable embedding model.
3. **Vector Database Setup**: Set up a vector database to store the embeddings.

<img width="1536" height="1024" alt="Image" src="https://github.com/user-attachments/assets/a231b6ad-80d7-4cf1-9b81-4fbd587f199e" />

**1. Data Collection and Splitting**
<img width="469" height="596" alt="Image" src="https://github.com/user-attachments/assets/736dfd7d-0b3e-4190-9b77-d936fc3e3290" />

#### for file loading
```
from langchain_community.document_loaders import TextLoader
f = TextLoader('../data/text_file/Python.txt',encoding='utf-8')
doc = f.load()
```
#### for Directry loading
```
from langchain_community.document_loaders import DirectoryLoader

dirload  = DirectoryLoader(
    path='../data/text_file',
    loader_cls=TextLoader,
    loader_kwargs={'encoding':'utf-8'},
    show_progress=True
)
dir_docs = dirload.load()

```
#### for data splitting
```
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(dir_docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")
```

#### for embedding
```
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

#Intializing the Embedding 
embeddingManager = EmbeddingManager()
embedding = embeddingManager.generate_Embedding(chunks)
```

#### for vectordb
```
class VectorStore:
    def __init__(self,collection_name:str="pdf_documents",persist_directory:str="../data/vector_store"):
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
    
    def add_documents(self,documents: list[any],embedding:np.ndarray):
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

vectordb = VectorStore()
vectordb.add_documents(all_splits,embedding)
``` 

## Process 2 : Retrieval and generation
### Process 2.1 : Retrieval

