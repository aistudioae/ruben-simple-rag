"""
Pinecone RAG Application
A simple Retrieval-Augmented Generation system using Pinecone vector database
"""

import os
import streamlit as st
from dotenv import load_dotenv
import openai
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
import tempfile
from typing import List, Dict, Any
import uuid
import logging
import time
from datetime import datetime

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
# Force load .env.pinecone, overriding any existing .env file
load_dotenv('.env.pinecone', override=True)

# Initialize clients
@st.cache_resource
def init_clients():
    """Initialize Pinecone and OpenAI clients"""
    # Initialize Pinecone
    api_key = os.getenv('PINECONE_API_KEY')
    logger.info(f"Pinecone API Key present: {bool(api_key)}")
    logger.info(f"Pinecone API Key length: {len(api_key) if api_key else 0}")
    
    pc = Pinecone(api_key=api_key)
    
    # Get or create index
    index_name = os.getenv('PINECONE_INDEX_NAME', 'ruben-rag')
    
    # Check if index exists
    if index_name not in pc.list_indexes().names():
        logger.info(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI text-embedding-3-small dimension
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    
    # Connect to index
    index = pc.Index(index_name)
    
    # Initialize OpenAI
    openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    return index, openai_client

def load_document(file) -> List[str]:
    """Load and chunk document based on file type"""
    start_time = time.time()
    logger.info(f"Loading document: {file.name} (size: {file.size:,} bytes)")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Load document based on file type
        load_start = time.time()
        if file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_path)
        elif file.name.endswith('.txt'):
            loader = TextLoader(tmp_path)
        elif file.name.endswith('.docx'):
            loader = Docx2txtLoader(tmp_path)
        else:
            raise ValueError(f"Unsupported file type: {file.name}")
        
        documents = loader.load()
        load_time = time.time() - load_start
        
        # Calculate total document length
        total_chars = sum(len(doc.page_content) for doc in documents)
        logger.info(f"Loaded document in {load_time:.2f}s - Total characters: {total_chars:,}")
        
        # Split documents into chunks
        chunk_start = time.time()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        chunk_time = time.time() - chunk_start
        
        # Log chunking details
        logger.info(f"Chunking completed in {chunk_time:.2f}s")
        logger.info(f"Created {len(chunks)} chunks from {total_chars:,} characters")
        logger.info(f"Average chunk size: {total_chars/len(chunks):.0f} characters")
        
        # Log first few chunks for debugging
        for i, chunk in enumerate(chunks[:3]):
            logger.debug(f"Chunk {i}: {len(chunk.page_content)} chars - Preview: {chunk.page_content[:100]}...")
        
        total_time = time.time() - start_time
        logger.info(f"Total document processing time: {total_time:.2f}s")
        
        return chunks
    
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)

def generate_embeddings(text: str, openai_client) -> List[float]:
    """Generate embeddings using OpenAI"""
    start_time = time.time()
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding_time = time.time() - start_time
    logger.debug(f"Generated embedding in {embedding_time:.3f}s for {len(text)} chars")
    return response.data[0].embedding

def store_in_pinecone(chunks: List, index, openai_client, filename: str):
    """Store document chunks in Pinecone"""
    start_time = time.time()
    logger.info(f"Starting to store {len(chunks)} chunks in Pinecone for {filename}")
    
    # Prepare vectors for upsert
    vectors = []
    embedding_times = []
    
    for i, chunk in enumerate(chunks):
        # Generate unique ID
        chunk_id = f"{filename}_{i}_{uuid.uuid4().hex[:8]}"
        
        # Generate embedding
        embed_start = time.time()
        embedding = generate_embeddings(chunk.page_content, openai_client)
        embed_time = time.time() - embed_start
        embedding_times.append(embed_time)
        
        # Prepare metadata
        metadata = {
            "content": chunk.page_content,
            "filename": filename,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunk_size": len(chunk.page_content)
        }
        
        vectors.append({
            "id": chunk_id,
            "values": embedding,
            "metadata": metadata
        })
        
        if (i + 1) % 5 == 0:
            logger.info(f"Processed {i + 1}/{len(chunks)} chunks...")
    
    # Log embedding generation stats
    total_embed_time = sum(embedding_times)
    avg_embed_time = total_embed_time / len(embedding_times) if embedding_times else 0
    logger.info(f"Embedding generation complete: Total time: {total_embed_time:.2f}s, Avg per chunk: {avg_embed_time:.3f}s")
    
    # Upsert to Pinecone in batches
    upsert_start = time.time()
    batch_size = 100
    total_upserted = 0
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        batch_start = time.time()
        index.upsert(vectors=batch)
        batch_time = time.time() - batch_start
        total_upserted += len(batch)
        logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size} ({len(batch)} vectors) in {batch_time:.2f}s")
    
    upsert_time = time.time() - upsert_start
    total_time = time.time() - start_time
    
    logger.info(f"Storage complete! Total vectors: {total_upserted}")
    logger.info(f"Time breakdown - Embeddings: {total_embed_time:.2f}s, Upsert: {upsert_time:.2f}s, Total: {total_time:.2f}s")

def semantic_search(query: str, index, openai_client, top_k: int = 5) -> List[Dict[str, Any]]:
    """Perform semantic search in Pinecone"""
    start_time = time.time()
    logger.info(f"Starting semantic search for query: '{query}' (length: {len(query)} chars)")
    
    # Generate query embedding
    embed_start = time.time()
    query_embedding = generate_embeddings(query, openai_client)
    embed_time = time.time() - embed_start
    logger.info(f"Query embedding generated in {embed_time:.3f}s")
    
    # Search in Pinecone
    search_start = time.time()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    search_time = time.time() - search_start
    
    # Log search results details
    logger.info(f"Pinecone search completed in {search_time:.3f}s")
    logger.info(f"Found {len(results.matches)} relevant documents (requested top_k={top_k})")
    
    # Log details about each match
    for i, match in enumerate(results.matches):
        logger.info(f"Match {i+1}: Score={match.score:.4f}, File={match.metadata.get('filename', 'Unknown')}, "
                   f"Chunk={match.metadata.get('chunk_index', '?')}/{match.metadata.get('total_chunks', '?')}, "
                   f"Size={match.metadata.get('chunk_size', '?')} chars")
    
    total_time = time.time() - start_time
    logger.info(f"Total search time: {total_time:.3f}s (embedding: {embed_time:.3f}s, search: {search_time:.3f}s)")
    
    return results.matches

def generate_response(query: str, context: List[Dict[str, Any]], openai_client) -> str:
    """Generate response using OpenAI with retrieved context"""
    start_time = time.time()
    
    # Prepare context from search results
    context_text = "\n\n".join([
        f"Document {i+1} (from {match.metadata.get('filename', 'Unknown')}):\n{match.metadata.get('content', '')}"
        for i, match in enumerate(context)
    ])
    
    total_context_chars = len(context_text)
    logger.info(f"Generating response with {len(context)} context chunks, total {total_context_chars:,} chars")
    
    # Create prompt
    system_prompt = """You are a helpful AI assistant. Use the provided context to answer questions. 
    If the answer cannot be found in the context, say so. Be concise and accurate."""
    
    user_prompt = f"""Context:
{context_text}

Question: {query}

Answer:"""
    
    # Log token estimation
    total_prompt_chars = len(system_prompt) + len(user_prompt)
    estimated_tokens = total_prompt_chars / 4  # Rough estimate: 1 token ≈ 4 chars
    logger.info(f"Prompt size: {total_prompt_chars:,} chars (≈{estimated_tokens:.0f} tokens)")
    
    # Generate response
    llm_start = time.time()
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    llm_time = time.time() - llm_start
    
    answer = response.choices[0].message.content
    total_time = time.time() - start_time
    
    logger.info(f"LLM response generated in {llm_time:.3f}s")
    logger.info(f"Response length: {len(answer)} chars")
    logger.info(f"Total response generation time: {total_time:.3f}s")
    
    return answer

# Streamlit UI
def main():
    st.set_page_config(page_title="Pinecone RAG", page_icon="🌲", layout="wide")
    
    st.title("🌲 Pinecone RAG Application")
    st.markdown("Upload documents and ask questions using AI-powered search")
    
    # Initialize clients
    try:
        index, openai_client = init_clients()
        st.success("✅ Connected to Pinecone and OpenAI")
    except Exception as e:
        st.error(f"Failed to initialize clients: {str(e)}")
        st.stop()
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'docx'],
            help="Upload a document to add to the knowledge base"
        )
        
        if uploaded_file is not None:
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    try:
                        # Load and chunk document
                        chunks = load_document(uploaded_file)
                        
                        # Store in Pinecone
                        store_in_pinecone(chunks, index, openai_client, uploaded_file.name)
                        
                        st.success(f"✅ Successfully processed {uploaded_file.name}")
                        st.info(f"Added {len(chunks)} chunks to the knowledge base")
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
        
        # Index statistics
        st.divider()
        st.header("📊 Index Statistics")
        try:
            stats = index.describe_index_stats()
            st.metric("Total Vectors", stats.total_vector_count)
            st.metric("Dimension", stats.dimension)
        except Exception as e:
            st.error(f"Could not fetch index stats: {str(e)}")
    
    # Main chat interface
    st.header("💬 Ask Questions")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating response..."):
                try:
                    # Perform semantic search
                    search_results = semantic_search(prompt, index, openai_client)
                    
                    if search_results:
                        # Generate response with context
                        response = generate_response(prompt, search_results, openai_client)
                        st.markdown(response)
                        
                        # Show sources in expander
                        with st.expander("📚 Sources"):
                            for i, match in enumerate(search_results):
                                st.markdown(f"**Source {i+1}** (Score: {match.score:.3f})")
                                st.markdown(f"*File: {match.metadata.get('filename', 'Unknown')}*")
                                st.markdown(match.metadata.get('content', '')[:300] + "...")
                                st.divider()
                    else:
                        response = "I couldn't find any relevant information in the knowledge base."
                        st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main() 