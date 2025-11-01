# 🎤 Frappe LLM

A voice-powered RAG (Retrieval-Augmented Generation) system that lets you query PDF documents using natural language speech. Built with Whisper for speech recognition, LangChain for RAG, and Ollama for local LLM inference.

## ✨ Features

- 🎙️ **Voice Input**: Speak your questions naturally using your microphone
- 📚 **PDF Document Query**: Load and query multiple PDF documents
- 🧠 **RAG System**: Retrieval-Augmented Generation for accurate, context-aware answers
- 🔒 **Local Processing**: All processing happens locally using Ollama
- ⚡ **GPU Support**: Automatic CUDA acceleration if available
- 💾 **Vector Database**: Persistent ChromaDB for fast document retrieval

## 🚀 Prerequisites

### 1. Install Ollama
Download and install [Ollama](https://ollama.ai/) for your system.

### 2. Pull Required Models
```bash
# Pull the LLM model (8B parameters)
ollama pull llama3.1:8b

# Pull the embedding model
ollama pull mxbai-embed-large
```

### 3. Python Requirements
- Python 3.8+
- CUDA-compatible GPU (optional, but recommended for faster processing)

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FrappeLLM.git
cd FrappeLLM
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install PyAudio (platform-specific):

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Linux:**
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

## 📂 Setup

1. Create a `data` folder in the project directory:
```bash
mkdir data
```

2. Add your PDF documents to the `data` folder:
```bash
cp your-documents.pdf data/
```

3. The system will automatically:
   - Create `ses_data` folder for audio recordings
   - Create `db` folder for the vector database
   - Process PDFs on first run (this may take a few minutes)

## 🎯 Usage

1. Start the application:
```bash
python frappeLLM.py
```

2. Wait for the system to load (first run takes longer):
```
============================================================
                    Frappe LLM
============================================================
   3 PDFs found:
     - document1.pdf
     - document2.pdf
     - document3.pdf
   Database found, loading...

============================================================
Ready!
            SPACE = Start/Stop recording | ESC = Exit
============================================================
```

3. Controls:
   - **SPACE**: Press to start recording, speak your question, press again to stop
   - **ESC**: Exit the application

4. Example workflow:
```
🎤 Speak...
  Recording stopped, processing...
  Transcribing with Whisper...

📝 You: What is the main topic of the document?

🤔 RAG system thinking...

🤖 Assistant: Based on the documents, the main topic is...
```

## 🛠️ Configuration

You can modify these settings in `frappeLLM.py`:

```python
# Whisper model size (tiny, base, small, medium, large)
whisper_model = whisper.load_model("medium", device=device)

# LLM model
llm = ChatOllama(model="llama3.1:8b")

# Embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Audio settings
RATE = 16000  # Sample rate
CHANNELS = 1  # Mono audio

# RAG settings
chunk_size=1000      # Text chunk size
chunk_overlap=200    # Overlap between chunks
k=4                  # Number of documents to retrieve
```

## 📋 Requirements

- `openai-whisper`: Speech-to-text transcription
- `torch`: Deep learning framework
- `pyaudio`: Audio recording
- `pynput`: Keyboard input handling
- `langchain-ollama`: Ollama integration
- `langchain-community`: Document loaders and vector stores
- `langchain-text-splitters`: Text chunking
- `pypdf`: PDF processing
- `chromadb`: Vector database

## 🐛 Troubleshooting

### No PDFs found error
- Ensure you have PDF files in the `data` folder
- Check that the files have `.pdf` extension

### PyAudio installation fails
- See platform-specific installation instructions above
- On Windows, try using `pipwin` instead of `pip`

### CUDA out of memory
- Reduce Whisper model size to `small` or `base`
- Consider using CPU instead (automatic fallback)

### Ollama connection error
- Ensure Ollama is running: `ollama serve`
- Check that models are downloaded: `ollama list`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [LangChain](https://github.com/langchain-ai/langchain) for RAG framework
- [Ollama](https://ollama.ai/) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

Made with ❤️ using Python and AI

