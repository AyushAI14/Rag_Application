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
### Implementation Steps
1. **Data Collection**: Gather relevant data from the organization's internal systems.
2. **Data Splitting**: Converting data into chunks.
2. **Embedding Generation**: Generate embeddings for the collected data using a suitable embedding model.
3. **Vector Database Setup**: Set up a vector database to store the embeddings.

<img width="1536" height="1024" alt="Image" src="https://github.com/user-attachments/assets/a231b6ad-80d7-4cf1-9b81-4fbd587f199e" />


#### Langchain Document 

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

<img width="1536" height="1024" alt="Image" src="https://github.com/user-attachments/assets/984dd9f9-0974-4d4f-ab55-9a7426f625d3" />

### Process 2.1 : Retrieval
```
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
rm = RetreivalManager(vectordb, embeddingManager)
```

### Process 2.2 : Generation

```
class GeminiLLM:
    def __init__(self,model_name:str="gemini-2.5-flash",api_key:str=None):
        """
        Intializing Gemini 
        Arg:
            model_name : This contain which model to use
            api_key : This will take gemini api_key
        """
        self.model_name = model_name
        self.api_key=api_key or os.getenv('GEMINI_API')
        if not self.api_key:
            print("Api key is not feeding to the LLM")
        self.llm = ChatGoogleGenerativeAI(
            api_key=self.api_key,
            model=self.model_name,
            max_output_tokens=1024,
            temperature=0.5
        )

        print(f"Gemini has been intialized, for model name {self.model_name} ")

        ## 2. Simple RAG function: retrieve context + generate response
    def rag_simple(self,query,retriever,top_k=3):
        ## retriever the context
        results=retriever.retreive(query,top_k=top_k)
        context="\n\n".join([doc['content'] for doc in results]) if results else ""
        if not context:
            return "No relevant context found to answer the question."
        
        ## generate the answwer using GROQ LLM
        prompt=f"""Use the following context to answer the question concisely.
            Context:
            {context}

            Question: {query}

            Answer:"""
        
        response = self.llm.invoke([prompt])

        return response.content
r = GeminiLLM()
```
**Output**
Outputs are quite crazy it really reading the context and giving the output base on pdf i feed
```
r.rag_simple("what is this laptop DELL Vostro Core i3 10th Gen , and give me more info about it",rm)
'This is a **DELL Vostro Core i3 10th Gen - (8 GB/512 GB SSD/Windows 10) Vostro 3401 Thin and Light Laptop**.\n\nMore info:\n*   **Model:** Vostro 3401\n*   **Processor:** Core i3 10th Gen\n*   **RAM:** 8 GB\n*   **Storage:** 512 GB SSD\n*   **Operating System:** Windows 10\n*   **Type:** Thin and Light Laptop\n*   **Warranty:** 1 Year Onsite Warranty\n*   **HSN/SAC:** 84713010\n*   **Serial No:** 1NBSYH3\n*   **Gross Price:** ₹35990.00\n*   **Discount:** ₹2500.00\n*   **Taxable Value:** ₹28381.36\n*   **IGST (18.0%):** ₹5108.64\n*   **Final Price (after discount & tax):** ₹33490.00'
```