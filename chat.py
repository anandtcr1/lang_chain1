# main_updated.py
import os
import sys
import warnings
from pathlib import Path

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# First check and install required packages
def check_and_install_packages():
    required = [
        'sentence_transformers',
        'transformers',
        'torch',
        'faiss',
        'llama_cpp',
        'langchain',
        'langchain_community'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"✗ {package}")
    
    if missing:
        print(f"\nMissing packages: {missing}")
        print("Please run: pip install " + " ".join(missing))
        return False
    return True

if not check_and_install_packages():
    sys.exit(1)

print("\nAll packages installed successfully!\n")

# Now import after verification
from dotenv import load_dotenv
load_dotenv()

# Create sample data if not exists
DATA_FILE = "data_source.txt"
if not os.path.exists(DATA_FILE):
    print(f"Creating sample {DATA_FILE}...")
    sample_data = """Artificial Intelligence (AI) is intelligence demonstrated by machines.
Machine Learning is a subset of AI that enables computers to learn from data.
Deep Learning uses neural networks with many layers.
Natural Language Processing allows computers to understand human language.
LangChain is a framework for building LLM-powered applications.
Vector databases like FAISS store and search embeddings efficiently.
Large Language Models (LLMs) are AI models trained on vast amounts of text data.
Embeddings are numerical representations of text that capture semantic meaning.
RAG (Retrieval Augmented Generation) combines retrieval and generation for better answers."""
    
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        f.write(sample_data)
    print(f"Created {DATA_FILE} with sample content")

# Create models directory
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

def download_model(url, filename):
    """Download a model from URL"""
    import requests
    from tqdm import tqdm
    
    model_path = os.path.join(MODEL_DIR, filename)
    
    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f, tqdm(
            desc="Progress",
            total=total_size,
            unit='iB',
            unit_scale=True,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"Model downloaded to {model_path}")
        return model_path
        
    except Exception as e:
        print(f"Download failed: {e}")
        return None

def main():
    print("="*60)
    print("DOCUMENT CHATBOT WITH LOCAL MODELS")
    print("="*60)
    
    # Step 1: Load and process documents
    print("\n1. Loading and processing documents...")
    try:
        from langchain_community.document_loaders import TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        loader = TextLoader(DATA_FILE, encoding='utf-8')
        documents = loader.load()
        print(f"   Loaded {len(documents)} document(s)")
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"   Split into {len(chunks)} chunks")
        
    except Exception as e:
        print(f"   Error loading documents: {e}")
        return
    
    # Step 2: Create embeddings
    print("\n2. Creating embeddings...")
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print(f"   Embeddings created successfully")
        
    except Exception as e:
        print(f"   Error creating embeddings: {e}")
        return
    
    # Step 3: Create vector store
    print("\n3. Creating vector store...")
    try:
        from langchain_community.vectorstores import FAISS
        
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("faiss_index")
        print("   Vector store created and saved")
        
    except Exception as e:
        print(f"   Error creating vector store: {e}")
        return
    
    # Step 4: Load or download LLM
    print("\n4. Setting up LLM...")
    model_path = os.getenv('MODEL_PATH')
    
    if not model_path or not os.path.exists(model_path):
        print("   No local model found.")
        print("\n   Available options:")
        print("   [1] TinyLlama-1.1B (500MB) - Fast, good for testing")
        print("   [2] Phi-2 (1.6GB) - Good quality, moderate size")
        print("   [3] Llama-2-7B (4GB) - Better quality, needs more RAM")
        print("   [4] Custom model path")
        
        choice = input("\n   Select option (1-4): ").strip()
        
        if choice == '1':
            url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
            filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
            model_path = download_model(url, filename)
            
        elif choice == '2':
            url = "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
            filename = "phi-2.Q4_K_M.gguf"
            model_path = download_model(url, filename)
            
        elif choice == '3':
            url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
            filename = "llama-2-7b-chat.Q4_K_M.gguf"
            model_path = download_model(url, filename)
            
        elif choice == '4':
            custom_path = input("   Enter full path to your GGUF model: ").strip()
            if os.path.exists(custom_path):
                model_path = custom_path
            else:
                print(f"   Model not found at {custom_path}")
                return
        else:
            print("   Invalid choice")
            return
        
        if not model_path:
            print("   Could not get model. Using simple search instead.")
            model_path = None
    
    # Load LLM if available
    llm = None
    if model_path and os.path.exists(model_path):
        try:
            from langchain_community.llms import LlamaCpp
            
            llm = LlamaCpp(
                model_path=model_path,
                temperature=0.7,
                max_tokens=512,
                top_p=0.95,
                n_ctx=2048,
                n_batch=512,
                verbose=False,
                streaming=False,
            )
            print("   LLM loaded successfully!")
            
        except Exception as e:
            print(f"   Error loading LLM: {e}")
            llm = None
    
    # Step 5: Setup chat system
    print("\n5. Setting up chat system...")
    try:
        from langchain.memory import ConversationBufferMemory
        from langchain.chains import ConversationalRetrievalChain
        
        # Load vector store
        try:
            vectorstore = FAISS.load_local("faiss_index", embeddings)
        except:
            # If can't load, create fresh
            vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Setup memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        # Setup retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # If LLM is available, use it. Otherwise use simple retriever.
        if llm:
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True,
                verbose=False
            )
            print("   Chat system with LLM ready!")
        else:
            # Create a simple QA chain without LLM
            from langchain.chains import RetrievalQA
            from langchain.prompts import PromptTemplate
            
            # Simple prompt template
            prompt_template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Context: {context}

            RULES
            - DO NOT ASK QUESTION
            - GIVE PRECISE ANSWER
            

            Question: {question}
            Answer: """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=None,  # No LLM, just retrieval
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            print("   Chat system with simple retrieval ready!")
        
    except Exception as e:
        print(f"   Error setting up chat system: {e}")
        return
    
    # Step 6: Chat interface
    print("\n" + "="*60)
    print("CHAT INTERFACE")
    print("="*60)
    print("Ask questions about your document.")
    print("Commands: 'quit' to exit, 'clear' to clear memory, 'sources' to toggle source display")
    print("="*60)
    
    show_sources = False
    
    while True:
        try:
            question = input("\nYou: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if question.lower() == 'clear':
                memory.clear()
                print("Memory cleared!")
                continue
            
            if question.lower() == 'sources':
                show_sources = not show_sources
                status = "ON" if show_sources else "OFF"
                print(f"Source display: {status}")
                continue
            
            if not question:
                continue
            
            print("Thinking...", end='', flush=True)
            
            # Get response - using invoke() instead of __call__()
            try:
                # Try invoke() method (new API)
                response = qa_chain.invoke({"question": question})
            except:
                # Fallback to __call__() for older versions
                response = qa_chain({"question": question})
            
            answer = response.get('answer', 'No answer generated.')
            sources = response.get('source_documents', [])
            
            print("\r" + " "*20 + "\r", end='')  # Clear "Thinking..."
            print(f"\nBot: {answer}")
            
            # Show sources if enabled
            if show_sources and sources:
                print("\n" + "-"*40)
                print("Sources used:")
                for i, source in enumerate(sources, 1):
                    content = source.page_content[:150].replace('\n', ' ')
                    print(f"{i}. {content}...")
                print("-"*40)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()