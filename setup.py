from setuptools import setup, find_packages

setup(
    name="rag_app",
    version="0.1.0",
    description="RAG Application pipeline",
    author="Ayush Vishwakarma",
    packages=find_packages(where="."),   # Automatically finds your src/ modules
    install_requires=[
        "langchain",
        "langchain-core",
        "langchain-community",
        "pypdf",
        "pymupdf",
        "sentence-transformers",
        "faiss-cpu",
        "chromadb",
        "python-dotenv",
    ],
    python_requires=">=3.9",
)
