# What is RAG Applications?

A way to optimize LLMs so they gain experience from an organization’s internal data, beyond their training data.

---

## Why RAG?

- Extends the scope of an LLM on a specific domain  
- Cost-efficient  
- More accurate  

---

## Solutions that RAG Provides

### Problem 1
Taking a recent model **GPT-5** which was launched in August — it is trained only on data up to that date.  
If we try to ask a question related to **September**, it may start hallucinating (making things up).  

### Problem 2
Consider a startup that has **policies, finance, HR data** as confidential information and wants to build a chatbot.  

- Fine-tuning a model would be **expensive and time-consuming**  
- Data would keep **updating continuously**  

---

## So, here comes RAG

- Accurate  
- Updates every time  
- Merges with LLM to provide better results  

---

## Working of RAG

<img width="1280" height="853" alt="RAG Workflow" src="https://github.com/user-attachments/assets/eb1b3a52-a804-4f6b-9055-7610244d272b" />


## Implementation of RAG
### Process 1
#### Implementation Steps
1. **Data Collection**: Gather relevant data from the organization's internal systems.
2. **Data Splitting**: Converting data into chunks.
2. **Embedding Generation**: Generate embeddings for the collected data using a suitable embedding model.
3. **Vector Database Setup**: Set up a vector database to store the embeddings.

<img width="1536" height="1024" alt="Image" src="https://github.com/user-attachments/assets/a231b6ad-80d7-4cf1-9b81-4fbd587f199e" />