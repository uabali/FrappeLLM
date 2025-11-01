import whisper
import torch
import pyaudio
import wave
import os
import glob
import threading
import time
from pynput import keyboard
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Directories
audio_folder = r"ses_data"
data_folder = r"data"
db_folder = r"db"

os.makedirs(audio_folder, exist_ok=True)
os.makedirs(data_folder, exist_ok=True)

print("=" * 60)
print("\t\tFrappe LLM")
print("=" * 60)

# Load Whisper model
# print("\n[1/3] Loading Whisper...")
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("medium", device=device)
#print(f"   Device: {device}")
#if device == "cuda":
#   print(f"   GPU: {torch.cuda.get_device_name(0)}")

# Load RAG system
#print("\n[2/3] Loading RAG system...")

# Find all PDFs in data/ folder
pdf_files = glob.glob(os.path.join(data_folder, "*.pdf"))

if not pdf_files:
    print(f"   WARNING: No PDFs found in {data_folder}!")
    print(f"   Please copy PDF files here.")
    exit(1)

print(f"   {len(pdf_files)} PDFs found:")
for pdf in pdf_files:
    print(f"     - {os.path.basename(pdf)}")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Load database if exists, otherwise create it
if os.path.exists(db_folder) and os.listdir(db_folder):
    print("   Database found, loading...")
    vector_store = Chroma(persist_directory=db_folder, embedding_function=embeddings)
else:
    print("   Creating database (may take a while first time)...")
    all_docs = []
    
    for pdf_path in pdf_files:
        print(f"   Loading: {os.path.basename(pdf_path)}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        all_docs.extend(docs)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(all_docs)
    print(f"   Total {len(splits)} document chunks created")
    
    vector_store = Chroma.from_documents(splits, embeddings, persist_directory=db_folder)
    print("   Database saved!")

# Load LLM
#print("\n[3/3] Loading LLM...")
llm = ChatOllama(model="llama3.1:8b")
#print("   Llama 3.1 8B ready")

print("\n" + "=" * 60)
print("Ready!")
print("\t\tSPACE = Start/Stop recording | ESC = Exit")
print("=" * 60 + "\n")

# PyAudio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
p = pyaudio.PyAudio()

recording_active = False
frames = []
stream = None

def query_rag(question):
    """Query RAG system"""
    docs = vector_store.similarity_search(question, k=4)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    answer = llm.invoke(prompt)
    return answer.content

def on_press(key):
    global recording_active, frames, stream
    
    try:
        if key == keyboard.Key.space:
            if not recording_active:
                # Start recording
                recording_active = True
                frames = []
                print("🎤 Speak...")
                
                stream = p.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK
                )
                
                def record_thread():
                    while recording_active:
                        data = stream.read(CHUNK)
                        frames.append(data)
                
                thread = threading.Thread(target=record_thread)
                thread.start()
                
            else:
                # Stop recording and process
                recording_active = False
                print("  Recording stopped, processing...")
                
                time.sleep(0.2)
                
                if stream:
                    stream.stop_stream()
                    stream.close()
                
                if len(frames) > 0:
                    # Delete previous audio files
                    for file in glob.glob(os.path.join(audio_folder, "*.wav")):
                        try:
                            os.remove(file)
                        except:
                            pass
                    
                    # Save audio
                    audio_file = os.path.join(audio_folder, "recording.wav")
                    
                    try:
                        wf = wave.open(audio_file, 'wb')
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(p.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        wf.writeframes(b''.join(frames))
                        wf.close()
                        
                        # Transcribe with Whisper
                        print(" Transcribing with Whisper...")
                        result = whisper_model.transcribe(audio_file, language="tr", fp16=False)
                        question = result['text'].strip()
                        
                        if question:
                            print(f"\ You: {question}")
                            
                            # Answer with RAG system
                            print("🤔 RAG system thinking...")
                            answer = query_rag(question)
                            print(f"\ Assistant: {answer}\n")
                        else:
                            print(" No audio detected!\n")
                            
                    except Exception as e:
                        print(f" Error: {e}\n")
                else:
                    print(" Could not save audio!\n")
        
        elif key == keyboard.Key.esc:
            print("\nExiting...")
            p.terminate()
            return False
            
    except AttributeError:
        pass

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()

