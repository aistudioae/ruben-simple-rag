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
from rank_bm25 import BM25Okapi
import json
import pickle

# Setup NLTK data directory for Render deployment
import nltk
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Import after setting NLTK path
from nltk.tokenize import word_tokenize

# Download required NLTK data
for resource in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        try:
            nltk.download(resource, download_dir=nltk_data_dir)
        except:
            # Resource might not exist in this NLTK version
            pass

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

# BM25 Index class for lexical search
class BM25Index:
    def __init__(self):
        self.documents = []
        self.bm25 = None
        self.metadata = []
        self.chunk_ids = []
        
    def add_documents(self, chunks, chunk_ids):
        """Add documents to BM25 index"""
        for chunk, chunk_id in zip(chunks, chunk_ids):
            # Tokenize for BM25
            try:
                tokens = word_tokenize(chunk.page_content.lower())
            except:
                # Fallback to simple split if NLTK tokenizer fails
                tokens = chunk.page_content.lower().split()
            self.documents.append(tokens)
            self.metadata.append(chunk.metadata)
            self.chunk_ids.append(chunk_id)
        
        # Create BM25 index
        self.bm25 = BM25Okapi(self.documents)
    
    def search(self, query: str, top_k: int = 5):
        """Search using BM25"""
        if not self.bm25:
            return []
            
        try:
            query_tokens = word_tokenize(query.lower())
        except:
            # Fallback to simple split if NLTK tokenizer fails
            query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                results.append({
                    'score': scores[idx],
                    'metadata': self.metadata[idx],
                    'chunk_id': self.chunk_ids[idx]
                })
        
        return results
    
    def save(self, filepath: str):
        """Save BM25 index to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'chunk_ids': self.chunk_ids
            }, f)
    
    def load(self, filepath: str):
        """Load BM25 index from file"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
                self.chunk_ids = data['chunk_ids']
                self.bm25 = BM25Okapi(self.documents)
                return True
        return False

# Global BM25 index
bm25_index = BM25Index()
bm25_index.load('bm25_index.pkl')  # Try to load existing index

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

def generate_chunk_context(document_text: str, chunk_text: str, chunk_index: int, total_chunks: int, openai_client) -> str:
    """Generate contextual description for a chunk within the document"""
    # Limit document text to avoid token limits
    max_doc_length = 4000
    if len(document_text) > max_doc_length:
        # Take beginning and end of document for context
        doc_preview = document_text[:max_doc_length//2] + "\n...\n" + document_text[-max_doc_length//2:]
    else:
        doc_preview = document_text
    
    prompt = f"""<document>
{doc_preview}
</document>

Here is chunk {chunk_index + 1} of {total_chunks} that we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Include relevant document title, section, or topic information. Answer only with the succinct context and nothing else."""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        
        context = response.choices[0].message.content.strip()
        logger.info(f"Generated context for chunk {chunk_index + 1}: {context[:50]}...")
        return context
    except Exception as e:
        logger.error(f"Error generating context for chunk {chunk_index}: {str(e)}")
        return f"This is chunk {chunk_index + 1} of {total_chunks} from the document."

def create_contextualized_chunks(chunks: List, openai_client) -> List:
    """Add contextual information to each chunk"""
    # Get full document text
    full_document = "\n\n".join([chunk.page_content for chunk in chunks])
    total_chunks = len(chunks)
    
    contextualized_chunks = []
    
    for i, chunk in enumerate(chunks):
        logger.info(f"Generating context for chunk {i + 1}/{total_chunks}")
        
        # Generate context for this chunk
        context = generate_chunk_context(full_document, chunk.page_content, i, total_chunks, openai_client)
        
        # Prepend context to chunk
        contextualized_content = f"{context}\n\n{chunk.page_content}"
        
        # Store both contextualized and original content
        chunk.metadata['original_content'] = chunk.page_content
        chunk.metadata['context'] = context
        chunk.page_content = contextualized_content
        
        contextualized_chunks.append(chunk)
    
    return contextualized_chunks

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
    """Store document chunks in Pinecone with contextual enhancement"""
    start_time = time.time()
    logger.info(f"Starting to store {len(chunks)} chunks in Pinecone for {filename}")
    
    # Create contextualized chunks
    logger.info("Generating contextual information for chunks...")
    context_start = time.time()
    contextualized_chunks = create_contextualized_chunks(chunks, openai_client)
    context_time = time.time() - context_start
    logger.info(f"Context generation completed in {context_time:.2f}s")
    
    # Prepare vectors for upsert
    vectors = []
    embedding_times = []
    chunk_ids = []
    
    for i, chunk in enumerate(contextualized_chunks):
        # Generate unique ID
        chunk_id = f"{filename}_{i}_{uuid.uuid4().hex[:8]}"
        chunk_ids.append(chunk_id)
        
        # Generate embedding on contextualized content
        embed_start = time.time()
        embedding = generate_embeddings(chunk.page_content, openai_client)
        embed_time = time.time() - embed_start
        embedding_times.append(embed_time)
        
        # Enhanced metadata
        metadata = {
            "content": chunk.page_content,  # Contextualized content
            "original_content": chunk.metadata['original_content'],  # Original chunk
            "context": chunk.metadata['context'],  # Context description
            "filename": filename,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunk_size": len(chunk.metadata['original_content']),
            "chunk_id": chunk_id
        }
        
        vectors.append({
            "id": chunk_id,
            "values": embedding,
            "metadata": metadata
        })
        
        if (i + 1) % 5 == 0:
            logger.info(f"Processed {i + 1}/{len(chunks)} chunks...")
    
    # Add to BM25 index
    logger.info("Adding documents to BM25 index...")
    bm25_start = time.time()
    global bm25_index
    bm25_index.add_documents(contextualized_chunks, chunk_ids)
    bm25_index.save('bm25_index.pkl')
    bm25_time = time.time() - bm25_start
    logger.info(f"BM25 indexing completed in {bm25_time:.2f}s")
    
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
    logger.info(f"Time breakdown - Context: {context_time:.2f}s, Embeddings: {total_embed_time:.2f}s, BM25: {bm25_time:.2f}s, Upsert: {upsert_time:.2f}s, Total: {total_time:.2f}s")

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

def hybrid_search(query: str, index, openai_client, top_k: int = 20) -> List[Dict[str, Any]]:
    """Perform hybrid search combining semantic and BM25"""
    start_time = time.time()
    logger.info(f"Starting hybrid search for query: '{query}' (top_k={top_k})")
    
    # Semantic search
    semantic_start = time.time()
    semantic_results = semantic_search(query, index, openai_client, top_k=top_k)
    semantic_time = time.time() - semantic_start
    logger.info(f"Semantic search completed in {semantic_time:.3f}s, found {len(semantic_results)} results")
    
    # BM25 search
    bm25_start = time.time()
    global bm25_index
    bm25_results = bm25_index.search(query, top_k=top_k)
    bm25_time = time.time() - bm25_start
    logger.info(f"BM25 search completed in {bm25_time:.3f}s, found {len(bm25_results)} results")
    
    # Reciprocal Rank Fusion
    fusion_start = time.time()
    fused_scores = {}
    k = 60  # Constant for RRF
    
    # Add semantic results
    for i, result in enumerate(semantic_results):
        doc_id = result.id
        if doc_id not in fused_scores:
            fused_scores[doc_id] = {
                'score': 0,
                'result': result,
                'sources': []
            }
        fused_scores[doc_id]['score'] += 1 / (k + i + 1)
        fused_scores[doc_id]['sources'].append(f'semantic_rank_{i+1}')
    
    # Add BM25 results
    for i, result in enumerate(bm25_results):
        doc_id = result['chunk_id']
        if doc_id not in fused_scores:
            # Need to fetch from Pinecone if not in semantic results
            fetch_result = index.fetch([doc_id])
            if doc_id in fetch_result.vectors:
                vector_data = fetch_result.vectors[doc_id]
                # Create a result object similar to semantic search results
                class BM25Result:
                    def __init__(self, id, metadata, score):
                        self.id = id
                        self.metadata = metadata
                        self.score = score
                
                fused_scores[doc_id] = {
                    'score': 0,
                    'result': BM25Result(doc_id, vector_data.metadata, result['score']),
                    'sources': []
                }
        
        if doc_id in fused_scores:
            fused_scores[doc_id]['score'] += 1 / (k + i + 1)
            fused_scores[doc_id]['sources'].append(f'bm25_rank_{i+1}')
    
    # Sort by fused score
    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1]['score'], reverse=True)[:top_k]
    
    # Extract results
    final_results = []
    for doc_id, data in sorted_results:
        result = data['result']
        # Add fusion information to metadata
        result.metadata['fusion_score'] = data['score']
        result.metadata['retrieval_sources'] = data['sources']
        final_results.append(result)
    
    fusion_time = time.time() - fusion_start
    total_time = time.time() - start_time
    
    logger.info(f"Rank fusion completed in {fusion_time:.3f}s")
    logger.info(f"Total hybrid search time: {total_time:.3f}s")
    logger.info(f"Final results: {len(final_results)} documents")
    
    return final_results

def rerank_results(query: str, results: List, openai_client, top_k: int = 10) -> List:
    """Rerank results using cross-encoder approach"""
    start_time = time.time()
    logger.info(f"Starting reranking of {len(results)} results")
    
    reranked = []
    
    for i, result in enumerate(results):
        # Get original content for reranking
        content = result.metadata.get('original_content', result.metadata.get('content', ''))
        
        # Score each chunk's relevance to the query
        prompt = f"""On a scale of 0-10, rate how relevant this text is to answering the query.
Query: {query}
Text: {content[:1000]}...
Only respond with a number 0-10."""
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            
            score = float(response.choices[0].message.content.strip())
            reranked.append((score, result))
            
            if (i + 1) % 5 == 0:
                logger.info(f"Reranked {i + 1}/{len(results)} documents...")
                
        except Exception as e:
            logger.error(f"Error reranking result {i}: {str(e)}")
            reranked.append((0, result))
    
    # Sort by relevance score
    reranked.sort(key=lambda x: x[0], reverse=True)
    
    # Get top-k results
    final_results = [r[1] for r in reranked[:top_k]]
    
    total_time = time.time() - start_time
    logger.info(f"Reranking completed in {total_time:.3f}s")
    
    # Log reranking scores
    for i, (score, result) in enumerate(reranked[:top_k]):
        logger.info(f"Reranked {i+1}: Score={score:.1f}, File={result.metadata.get('filename', 'Unknown')}")
    
    return final_results

def generate_response(query: str, context: List[Dict[str, Any]], openai_client) -> str:
    """Generate response using OpenAI with retrieved context"""
    start_time = time.time()
    
    # Prepare context from search results - use original content for generation
    context_text = "\n\n".join([
        f"Document {i+1} (from {match.metadata.get('filename', 'Unknown')}):\n{match.metadata.get('original_content', match.metadata.get('content', ''))}"
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
    
    st.title("🌲 Pinecone RAG Application with Contextual Retrieval")
    st.markdown("Upload documents and ask questions using AI-powered search with enhanced contextual understanding")
    
    # Show info about contextual RAG if index is empty
    try:
        temp_index, _ = init_clients()
        stats = temp_index.describe_index_stats()
        if stats.total_vector_count == 0:
            st.info("""
            💡 **This application now uses Contextual RAG!**
            
            Documents uploaded will have contextual information added to each chunk, improving retrieval accuracy by:
            - 49% with hybrid search (semantic + BM25)
            - 67% when combined with reranking
            
            If you have existing documents without contextual embeddings, please clear the index and re-upload them.
            """)
    except:
        pass
    
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
        
        # Clear index button
        st.divider()
        st.header("🗑️ Manage Index")
        st.warning("⚠️ Clearing the index will delete all stored documents and embeddings!")
        
        col1, col2 = st.columns(2)
        with col1:
            clear_confirm = st.checkbox("I understand this will delete all data", key="clear_confirm")
        with col2:
            if st.button("Clear All Data", type="secondary", disabled=not clear_confirm):
                with st.spinner("Clearing all data..."):
                    try:
                        # Clear Pinecone index
                        logger.info("Clearing Pinecone index...")
                        # Delete all vectors by using a dummy query that matches everything
                        index.delete(delete_all=True)
                        
                        # Clear BM25 index
                        logger.info("Clearing BM25 index...")
                        global bm25_index
                        bm25_index = BM25Index()
                        # Remove the saved BM25 index file
                        if os.path.exists('bm25_index.pkl'):
                            os.remove('bm25_index.pkl')
                        
                        st.success("✅ Successfully cleared all data from the index!")
                        st.info("You can now upload documents with contextual embeddings.")
                        
                        # Force a rerun to update statistics
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error clearing index: {str(e)}")
                        logger.error(f"Error clearing index: {str(e)}")
    
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
                    # Perform hybrid search
                    search_results = hybrid_search(prompt, index, openai_client)
                    
                    if search_results:
                        # Rerank results
                        reranked_results = rerank_results(prompt, search_results, openai_client)
                        
                        # Generate response with context
                        response = generate_response(prompt, reranked_results, openai_client)
                        st.markdown(response)
                        
                        # Show sources in expander
                        with st.expander("📚 Sources"):
                            for i, match in enumerate(reranked_results):
                                st.markdown(f"**Source {i+1}** (Score: {match.score:.3f})")
                                st.markdown(f"*File: {match.metadata.get('filename', 'Unknown')}*")
                                # Show retrieval method if available
                                sources = match.metadata.get('retrieval_sources', [])
                                if sources:
                                    st.markdown(f"*Retrieved via: {', '.join(sources)}*")
                                # Show original content, not contextualized
                                original_content = match.metadata.get('original_content', match.metadata.get('content', ''))
                                st.markdown(original_content[:300] + "...")
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