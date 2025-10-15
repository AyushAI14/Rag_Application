from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from src.vector_store.vector_store import VectorStore
from src.embeddings.emdedding import EmbeddingManager
from src.retrieval.retreive import RetreivalManager
# from langchain.prompts import PromptTemplate
# from langchain.schema import HumanMessage, SystemMessage
import os
load_dotenv()


class GeminiLLM:
    def __init__(self,model_name:str="gemini-2.5-flash-lite",api_key:str=None):
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
        print('LLM Is Generating ...')
        return response.content
r = GeminiLLM()

vectordb = VectorStore()
embeddingManager = EmbeddingManager()
rm = RetreivalManager(vectordb, embeddingManager)

# print(r.rag_simple("what is this laptop DELL Vostro Core i3 10th Gen , and give me more info about it",rm))