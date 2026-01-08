# # main_fixed.py
# import os
# import sys
# from pathlib import Path

# # First check and install required packages
# def check_and_install_packages():
#     required = [
#         'sentence_transformers',
#         'transformers',
#         'torch',
#         'faiss',
#         'llama_cpp',
#         'langchain',
#         'langchain_community'
#     ]
    
#     missing = []
#     for package in required:
#         try:
#             __import__(package.replace('-', '_'))
#             print(f"✓ {package}")
#         except ImportError:
#             missing.append(package)
#             print(f"✗ {package}")
    
#     if missing:
#         print(f"\nMissing packages: {missing}")
#         print("Please run: pip install " + " ".join(missing))
#         return False
#     return True

# if not check_and_install_packages():
#     sys.exit(1)

# print("\nAll packages installed successfully!\n")

# # Now import after verification
# from dotenv import load_dotenv
# load_dotenv()

# # Create sample data if not exists
# DATA_FILE = "data_source.txt"
# if not os.path.exists(DATA_FILE):
#     print(f"Creating sample {DATA_FILE}...")
#     sample_data = """Artificial Intelligence (AI) is intelligence demonstrated by machines.
# Machine Learning is a subset of AI that enables computers to learn from data.
# Deep Learning uses neural networks with many layers.
# Natural Language Processing allows computers to understand human language.
# LangChain is a framework for building LLM-powered applications.
# Vector databases like FAISS store and search embeddings efficiently."""
    
#     with open(DATA_FILE, 'w', encoding='utf-8') as f:
#         f.write(sample_data)
#     print(f"Created {DATA_FILE} with sample content")

# # Create models directory
# MODEL_DIR = "./models"
# os.makedirs(MODEL_DIR, exist_ok=True)

# def main():
#     print("="*60)
#     print("DOCUMENT CHATBOT WITH LOCAL MODELS")
#     print("="*60)
    
#     # Step 1: Load and process documents
#     print("\n1. Loading and processing documents...")
#     try:
#         from langchain_community.document_loaders import TextLoader
#         from langchain.text_splitter import CharacterTextSplitter
        
#         loader = TextLoader(DATA_FILE, encoding='utf-8')
#         documents = loader.load()
#         print(f"   Loaded {len(documents)} document(s)")
        
#         # Split into chunks
#         text_splitter = CharacterTextSplitter(
#             separator="\n",
#             chunk_size=300,
#             chunk_overlap=50,
#             length_function=len,
#             is_separator_regex=False,
#         )
#         chunks = text_splitter.split_documents(documents)
#         print(f"   Split into {len(chunks)} chunks")
        
#     except Exception as e:
#         print(f"   Error loading documents: {e}")
#         return
    
#     # Step 2: Create embeddings
#     print("\n2. Creating embeddings...")
#     try:
#         from langchain_community.embeddings import HuggingFaceEmbeddings
        
#         # Use a simple model that's easy to download
#         embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2",
#             model_kwargs={'device': 'cpu'},
#             encode_kwargs={'normalize_embeddings': False}
#         )
        
#         # Test embeddings
#         test_text = "Hello, world!"
#         test_embedding = embeddings.embed_query(test_text)
#         print(f"   Embeddings created successfully (dimension: {len(test_embedding)})")
        
#     except Exception as e:
#         print(f"   Error creating embeddings: {e}")
#         return
    
#     # Step 3: Create vector store
#     print("\n3. Creating vector store...")
#     try:
#         from langchain_community.vectorstores import FAISS
        
#         vectorstore = FAISS.from_documents(chunks, embeddings)
#         vectorstore.save_local("faiss_index")
#         print("   Vector store created and saved")
        
#     except Exception as e:
#         print(f"   Error creating vector store: {e}")
#         return
    
#     # Step 4: Load or download LLM
#     print("\n4. Setting up LLM...")
#     model_path = os.getenv('MODEL_PATH')
    
#     if not model_path or not os.path.exists(model_path):
#         print("   No local model found.")
#         print("   You need to download a GGUF model first.")
#         print("\n   Recommended models:")
#         print("   1. TinyLlama (500MB): https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
#         print("   2. Phi-2 (1.6GB): https://huggingface.co/TheBloke/phi-2-GGUF")
#         print("   3. Llama2-7B (4GB): https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF")
        
#         choice = input("\n   Enter model URL or local path: ").strip()
        
#         if choice.startswith('http'):
#             # Download model
#             import requests
#             from tqdm import tqdm
            
#             # Extract filename from URL
#             filename = choice.split('/')[-1]
#             model_path = os.path.join(MODEL_DIR, filename)
            
#             print(f"\n   Downloading {filename}...")
#             try:
#                 response = requests.get(choice, stream=True)
#                 total_size = int(response.headers.get('content-length', 0))
                
#                 with open(model_path, 'wb') as f, tqdm(
#                     desc="Progress",
#                     total=total_size,
#                     unit='iB',
#                     unit_scale=True,
#                 ) as pbar:
#                     for chunk in response.iter_content(chunk_size=8192):
#                         if chunk:
#                             f.write(chunk)
#                             pbar.update(len(chunk))
                
#                 print(f"   Model downloaded to {model_path}")
                
#                 # Save to .env file
#                 with open('.env', 'w') as f:
#                     f.write(f"MODEL_PATH={model_path}\n")
                
#             except Exception as e:
#                 print(f"   Download failed: {e}")
#                 print("\n   Please download manually and specify path.")
#                 return
#         else:
#             model_path = choice
#             if not os.path.exists(model_path):
#                 print(f"   Model not found at {model_path}")
#                 return
    
#     # Load the LLM
#     try:
#         from langchain_community.llms import LlamaCpp
        
#         llm = LlamaCpp(
#             model_path=model_path,
#             temperature=0.7,
#             max_tokens=512,
#             top_p=1,
#             n_ctx=2048,
#             n_batch=512,
#             verbose=False,
#         )
#         print("   LLM loaded successfully!")
        
#     except Exception as e:
#         print(f"   Error loading LLM: {e}")
#         print("\n   Continuing without LLM (using mock responses)...")
        
#         # Mock LLM for testing
#         class MockLLM:
#             def __call__(self, prompt):
#                 return "This is a mock response. The real LLM could not be loaded."
        
#         llm = MockLLM()
    
#     # Step 5: Create chat chain - FIXED VERSION
#     print("\n5. Setting up chat system...")
#     try:
#         from langchain.memory import ConversationBufferMemory
#         from langchain.chains import ConversationalRetrievalChain
        
#         # Load vector store - without allow_dangerous_deserialization parameter
#         try:
#             # Try with parameter for newer versions
#             vectorstore = FAISS.load_local(
#                 "faiss_index", 
#                 embeddings
#             )
#         except TypeError:
#             # If that fails, try without parameter
#             vectorstore = FAISS.load_local(
#                 "faiss_index", 
#                 embeddings
#             )
        
#         memory = ConversationBufferMemory(
#             memory_key="chat_history",
#             return_messages=True
#         )
        
#         qa_chain = ConversationalRetrievalChain.from_llm(
#             llm=llm,
#             retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
#             memory=memory,
#             verbose=False
#         )
        
#         print("   Chat system ready!")
        
#     except Exception as e:
#         print(f"   Error setting up chat: {e}")
#         print("\nTrying alternative approach...")
        
#         # Alternative: Create new vector store each time
#         try:
#             vectorstore = FAISS.from_documents(chunks, embeddings)
            
#             memory = ConversationBufferMemory(
#                 memory_key="chat_history",
#                 return_messages=True
#             )
            
#             qa_chain = ConversationalRetrievalChain.from_llm(
#                 llm=llm,
#                 retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
#                 memory=memory,
#                 verbose=False
#             )
            
#             print("   Chat system ready (using fresh vector store)!")
#         except Exception as e2:
#             print(f"   Failed to set up chat system: {e2}")
#             return
    
#     # Step 6: Chat interface
#     print("\n" + "="*60)
#     print("CHAT INTERFACE")
#     print("="*60)
#     print("Type your questions about the document.")
#     print("Commands: 'quit' to exit, 'clear' to clear memory")
    
#     while True:
#         try:
#             question = input("\nYou: ").strip()
            
#             if question.lower() in ['quit', 'exit', 'q']:
#                 print("\nGoodbye!")
#                 break
            
#             if question.lower() == 'clear':
#                 memory.clear()
#                 print("Memory cleared!")
#                 continue
            
#             if not question:
#                 continue
            
#             print("Thinking...", end='', flush=True)
            
#             # Get response
#             response = qa_chain({"question": question})
#             answer = response.get('answer', 'No answer generated.')
            
#             print("\r" + " "*20 + "\r", end='')  # Clear "Thinking..."
#             print(f"Bot: {answer}")
            
#         except KeyboardInterrupt:
#             print("\n\nInterrupted. Goodbye!")
#             break
#         except Exception as e:
#             print(f"\nError: {str(e)}")

# if __name__ == "__main__":
#     main()

# minimal_chat.py
import os
import sys
from dotenv import load_dotenv

load_dotenv()

print("Initializing Document Chatbot...")

try:
    # Import all required modules
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
    print("✓ All imports successful!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Create or load data
DATA_FILE = "data_source.txt"
if not os.path.exists(DATA_FILE):
    print(f"Creating {DATA_FILE}...")
    with open(DATA_FILE, "w") as f:
        f.write("""Artificial Intelligence (AI) is the simulation of human intelligence.
Machine Learning allows computers to learn from data.
Deep Learning uses neural networks.
Natural Language Processing (NLP) helps computers understand language.
LangChain is a framework for LLM applications.""")

# 1. Load and split documents
print("\n1. Loading documents...")
loader = TextLoader(DATA_FILE)
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separator="\n"
)
chunks = text_splitter.split_documents(documents)
print(f"   Created {len(chunks)} chunks")

# 2. Create embeddings
print("\n2. Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Create vector store (don't save/load, create fresh each time)
print("\n3. Creating vector store...")
vectorstore = FAISS.from_documents(chunks, embeddings)
print("   Vector store created")

# 4. Setup LLM
print("\n4. Loading LLM...")
model_path = os.getenv('MODEL_PATH', './models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')

if not os.path.exists(model_path):
    print(f"   Model not found at {model_path}")
    print("   Using simple search-based responses")
    
    # Simple search-based response (no LLM)
    def simple_response(query):
        # Search for similar chunks
        docs = vectorstore.similarity_search(query, k=2)
        if docs:
            return f"Based on the document: {docs[0].page_content[:200]}..."
        return "I don't have enough information to answer that."
    
    llm = lambda x: simple_response(x)
else:
    try:
        from langchain_community.llms import LlamaCpp
        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.7,
            max_tokens=256,
            n_ctx=1024,
            verbose=False
        )
        print("   LLM loaded successfully!")
    except Exception as e:
        print(f"   Error loading LLM: {e}")
        llm = lambda x: f"Search result: {vectorstore.similarity_search(x, k=1)[0].page_content[:100]}..."

# 5. Setup conversation chain
print("\n5. Setting up conversation chain...")
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    memory=memory,
    verbose=False
)

print("\n" + "="*50)
print("CHATBOT READY! Ask questions about the document.")
print("Type 'quit' to exit, 'clear' to clear memory")
print("="*50)

# Chat loop
while True:
    try:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            memory.clear()
            print("Memory cleared!")
            continue
        
        if not user_input:
            continue
        
        # Get response
        result = qa_chain({"question": user_input})
        print(f"\nBot: {result['answer']}")
        
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except Exception as e:
        print(f"\nError: {e}")