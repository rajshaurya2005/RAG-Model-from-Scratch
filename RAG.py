import json
from pathlib import Path
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

class SimpleRAG:
    def __init__(self, groq_api_key, embedding_model="all-MiniLM-L6-v2", chunk_size=500, chunk_overlap=50):
        self.groq = Groq(api_key=groq_api_key)
        self.embedder = SentenceTransformer(embedding_model)
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("docs")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _read_file(self, path):
        """Read text from PDF, TXT, or JSON files"""
        ext = Path(path).suffix.lower()
        if ext == '.pdf':
            with open(path, 'rb') as f:
                return ''.join(page.extract_text() for page in PyPDF2.PdfReader(f).pages)
        elif ext == '.txt':
            return Path(path).read_text(encoding='utf-8')
        elif ext == '.json':
            return json.dumps(json.loads(Path(path).read_text()), indent=2)
        return ""

    def _chunk_text(self, text):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            
            # Move start position considering overlap
            if end >= len(words):
                break
            start = end - self.chunk_overlap
            
        return chunks

    def add_document(self, file_path):
        """Add a document to the RAG system"""
        text = self._read_file(file_path)
        if not text:
            return 0
        
        chunks = self._chunk_text(text)
        embeddings = self.embedder.encode(chunks).tolist()
        ids = [f"doc_{i}" for i in range(len(chunks))]
        
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=ids
        )
        return len(chunks)

    def generate_response(self, question, model_name="llama3-8b-8192", temperature=0.1, max_tokens=1024, n_results=3):
        """Query the RAG system and get an AI response with customizable parameters"""
        query_embedding = self.embedder.encode([question]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        if not results['documents'][0]:
            return "No relevant information found."
        
        context = '\n\n'.join(results['documents'][0])
        
        prompt = f"""Answer the question based on this context:

Context:
{context}

Question: {question}

Answer:"""
        
        response = self.groq.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content


rag = SimpleRAG("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")
rag.add_document("/content/gachibowli.pdf")
answer = rag.generate_response("What is the recipe for deep fried fish cakes?")
print(answer)

rag = SimpleRAG(
    groq_api_key="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",
    embedding_model="all-mpnet-base-v2",  # Better but slower embedding model
    chunk_size=300,                       # Smaller chunks
    chunk_overlap=30                      # Less overlap
)

rag.add_document("document.pdf")

answer = rag.generate_response(
    "What is this document about?",
    model_name="mixtral-8x7b-32768",      # Different Groq model
    temperature=0.3,                      # More creative responses
    max_tokens=2048,                      # Longer responses
    n_results=5                           # More context chunks
)
