# SimpleRAG - Document Question Answering System

A lightweight Python class for building Retrieval-Augmented Generation (RAG) systems. Ask questions about your PDF, TXT, and JSON documents using AI.

## Quick Start

```python
from simple_rag import SimpleRAG

# Basic usage
rag = SimpleRAG("YOUR_GROQ_API_KEY")
rag.add_document("document.pdf")
answer = rag.generate_response("What is this document about?")
print(answer)
```

## Installation

```bash
pip install PyPDF2 chromadb sentence-transformers groq pathlib
```

## Features

- **Multi-format Support**: PDF, TXT, JSON files
- **Configurable Chunking**: Customizable text splitting with overlap
- **Vector Search**: ChromaDB for fast document retrieval
- **Multiple AI Models**: Access to Groq's language models
- **Simple API**: Easy to integrate and use

## Usage Examples

### Basic Document Q&A
```python
rag = SimpleRAG("YOUR_GROQ_API_KEY")
rag.add_document("document.pdf")
answer = rag.generate_response("What is this document about?")
print(answer)
```

### Advanced Configuration
```python
rag = SimpleRAG(
    groq_api_key="YOUR_GROQ_API_KEY",
    embedding_model="all-mpnet-base-v2",  # Better quality embeddings
    chunk_size=300,                       # Smaller text chunks
    chunk_overlap=30                      # Reduced overlap
)

rag.add_document("document.pdf")
answer = rag.generate_response(
    "What is this document about?",
    model_name="mixtral-8x7b-32768",      # Different AI model
    temperature=0.3,                      # More creative responses
    max_tokens=2048,                      # Longer responses
    n_results=5                           # More context chunks
)
```

## Configuration Options

### Initialization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `groq_api_key` | Required | Your Groq API key |
| `embedding_model` | `"all-MiniLM-L6-v2"` | SentenceTransformer model |
| `chunk_size` | `500` | Words per text chunk |
| `chunk_overlap` | `50` | Overlapping words between chunks |

### Response Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `"llama3-8b-8192"` | Groq model to use |
| `temperature` | `0.1` | Response creativity (0-1) |
| `max_tokens` | `1024` | Maximum response length |
| `n_results` | `3` | Number of context chunks |

## Available Models

**Groq Language Models:**
- `llama3-8b-8192` - Fast, general purpose (default)
- `mixtral-8x7b-32768` - More capable, longer context
- `llama3-70b-8192` - Most capable Llama model
- `gemma-7b-it` - Google's Gemma model

**Embedding Models:**
- `all-MiniLM-L6-v2` - Fast, good quality (default)
- `all-mpnet-base-v2` - Better quality, slower
- `all-distilroberta-v1` - Balanced option

## Supported File Types

| Format | Extension | Processing |
|--------|-----------|------------|
| PDF | `.pdf` | Text extraction from all pages |
| Text | `.txt` | UTF-8 encoded plain text |
| JSON | `.json` | Formatted JSON structure |

## API Reference

### `SimpleRAG(groq_api_key, embedding_model, chunk_size, chunk_overlap)`
Initialize the RAG system.

### `add_document(file_path)`
Add a document to the knowledge base.
- **Returns**: Number of chunks created
- **Returns**: `0` if document is empty or unreadable

### `generate_response(question, model_name, temperature, max_tokens, n_results)`
Generate an AI response based on document content.
- **Returns**: AI-generated answer string
- **Returns**: `"No relevant information found."` if no matching content

## Getting Started

1. **Get Groq API Key**: Sign up at [console.groq.com](https://console.groq.com)
2. **Install Dependencies**: Run the pip install command above
3. **Replace API Key**: Update `"YOUR_GROQ_API_KEY"` with your actual key
4. **Add Documents**: Use `add_document()` to load your files
5. **Ask Questions**: Use `generate_response()` to query your documents

## Error Notes

**Code Syntax Issues**: The provided code has syntax errors that need fixing:
- Change `**init**` to `__init__`
- Change `*read*file` to `_read_file`  
- Change `*chunk*text` to `_chunk_text`

**Corrected Method Names:**
```python
def __init__(self, ...):  # Constructor
def _read_file(self, ...):  # Private method
def _chunk_text(self, ...):  # Private method
```

## Performance Tips

- **Small chunks (200-400)** for precise answers
- **Large chunks (500-800)** for broader context  
- **Low temperature (0.1-0.3)** for factual responses
- **Higher temperature (0.3-0.7)** for creative answers
- **More n_results** if answers lack sufficient context

## Requirements

- Python 3.7+
- Groq API key
- Internet connection for API calls and model downloads

## Limitations

- ChromaDB storage is not persistent between sessions
- All documents stored in single collection
- Text-only processing (no images or complex formatting)
- Requires active internet connection

## Security Note

Never commit API keys to version control. Use environment variables or config files that are gitignored.

```python
import os
rag = SimpleRAG(os.getenv("GROQ_API_KEY"))
```
