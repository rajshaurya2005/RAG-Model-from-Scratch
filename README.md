impleRAG: A Lightweight Retrieval-Augmented Generation System
SimpleRAG is a Python class designed to demonstrate a basic Retrieval-Augmented Generation (RAG) system. It allows you to ingest documents (PDF, TXT, JSON), chunk them, create embeddings, store them in a ChromaDB vector store, and then use a Large Language Model (LLM) from Groq to answer questions based on the retrieved relevant information.

Features
Document Ingestion: Reads text from PDF, TXT, and JSON files.
Text Chunking: Splits documents into manageable, overlapping chunks for better retrieval.
Sentence Embeddings: Uses the sentence-transformers library to convert text chunks into numerical embeddings.
Vector Store: Utilizes chromadb for efficient storage and retrieval of document chunks.
LLM Integration: Connects with the Groq API to leverage powerful language models for generating answers.
Customizable Parameters: Allows you to configure embedding models, chunking strategy, and LLM parameters.
Installation
To use SimpleRAG, you'll need to install the necessary libraries:

Bash

pip install PyPDF2 chromadb sentence-transformers groq
Getting Started
1. Obtain a Groq API Key
You'll need an API key from Groq to use their language models. Get one from the Groq Console.

2. Initialize SimpleRAG
Python

from simple_rag import SimpleRAG # Assuming you save the class in simple_rag.py

# Initialize with your Groq API key
rag = SimpleRAG("your-groq-api-key")

# You can also customize parameters during initialization:
rag_custom = SimpleRAG(
    groq_api_key="your-groq-api-key",
    embedding_model="all-mpnet-base-v2",  # A more powerful embedding model
    chunk_size=300,                        # Smaller chunks
    chunk_overlap=30                       # Less overlap
)
3. Add Documents
You can add PDF, TXT, or JSON documents to your RAG system.

Python

# Add a PDF document
rag.add_document("path/to/your/document.pdf")

# Add a text document
rag.add_document("path/to/your/notes.txt")

# Add a JSON document
rag.add_document("path/to/your/data.json")
4. Generate Responses
Once documents are added, you can ask questions and get AI-generated answers based on the context.

Python

question = "What is this document about?"
answer = rag.generate_response(question)
print(answer)

# Customize response generation parameters:
answer_custom = rag_custom.generate_response(
    "What are the key findings discussed?",
    model_name="mixtral-8x7b-32768",      # Use a different Groq model
    temperature=0.3,                     # Adjust creativity (0.0 to 1.0)
    max_tokens=2048,                     # Control response length
    n_results=5                          # Retrieve more context chunks
)
print(answer_custom)
Class Reference
SimpleRAG(groq_api_key, embedding_model="all-MiniLM-L6-v2", chunk_size=500, chunk_overlap=50)
groq_api_key (str): Your API key for Groq.
embedding_model (str, optional): The name of the sentence-transformers model to use for generating embeddings. Defaults to "all-MiniLM-L6-v2".
chunk_size (int, optional): The maximum number of words in each text chunk. Defaults to 500.
chunk_overlap (int, optional): The number of words that overlap between consecutive chunks. Defaults to 50.
add_document(file_path)
file_path (str): The path to the document (PDF, TXT, or JSON) to add to the RAG system.
Returns: int - The number of chunks added from the document.
generate_response(question, model_name="llama3-8b-8192", temperature=0.1, max_tokens=1024, n_results=3)
question (str): The question to ask the RAG system.
model_name (str, optional): The name of the Groq model to use for generating the answer. Defaults to "llama3-8b-8192".
temperature (float, optional): Controls the randomness of the LLM's output. Higher values (e.g., 0.7) make the output more creative, while lower values (e.g., 0.1) make it more focused and deterministic. Defaults to 0.1.
max_tokens (int, optional): The maximum number of tokens (words/sub-words) the LLM can generate in its response. Defaults to 1024.
n_results (int, optional): The number of most relevant document chunks to retrieve from the vector store to provide as context to the LLM. Defaults to 3.
Returns: str - The AI-generated answer to the question.
