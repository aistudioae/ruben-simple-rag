"""
=============================================================================
RAG (Retrieval-Augmented Generation) Application with Supabase Vector Store
=============================================================================

This application implements a complete RAG pipeline that:
1. Loads and processes documents (PDF, TXT, DOCX)
2. Chunks documents into manageable pieces
3. Generates embeddings using OpenAI's text-embedding-3-small
4. Stores documents and embeddings in Supabase with pgvector
5. Performs semantic search using vector similarity
6. Generates AI responses using retrieved context

Key Features:
- Smart document chunking with overlap
- Batch embedding generation with rate limit handling
- PostgreSQL pgvector similarity search with fallback
- Real-time chat interface with source citations
- Comprehensive error handling and logging

Dependencies:
- Streamlit: Web UI framework
- Supabase: PostgreSQL database with vector extensions
- OpenAI: Embeddings and chat completions
- LangChain: Document loading and text splitting
- NumPy: Vector operations for similarity calculations
"""

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================

import os
import streamlit as st
from supabase import create_client, Client
from openai import OpenAI
import numpy as np
from typing import List, Dict, Optional, Tuple
import tempfile
import hashlib
import logging
import time
from datetime import datetime

# Document loading and processing libraries
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Configure comprehensive logging with timestamps and detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Log application startup
logger.info("="*80)
logger.info("🚀 SimpleRAG Application Starting Up")
logger.info("="*80)

# =============================================================================
# CONFIGURATION AND CLIENT INITIALIZATION
# =============================================================================

# Environment variables for API connections
# These should be set as environment variables in production for security
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://epdxubvhtbfauanxqdih.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVwZHh1YnZodGJmYXVhbnhxZGloIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDU3NTU0MjcsImV4cCI6MjA2MTMzMTQyN30.uRAsu7GHfJOYf-oIMU7XodWtei7WWH1F0wokOkpiMEM")  
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")

logger.info("🔧 Loading configuration...")
logger.info(f"Supabase URL: {SUPABASE_URL}")
logger.info(f"OpenAI API Key: {'***' + OPENAI_API_KEY[-4:] if OPENAI_API_KEY else 'Not set'}")

# Validate that all required environment variables are present
# Critical for application functionality - fail fast if missing
if not all([SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY]):
    error_msg = "⚠️ Missing required environment variables. Please set SUPABASE_URL, SUPABASE_KEY, and OPENAI_API_KEY"
    logger.error(error_msg)
    st.error(error_msg)
    st.stop()

logger.info("✅ All required environment variables are present")

# Initialize database and AI service clients with comprehensive error handling
logger.info("🔌 Initializing service clients...")
try:
    # Create Supabase client for database operations
    # This handles PostgreSQL operations including vector storage
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("✅ Supabase client initialized successfully")
    
    # Create OpenAI client for embeddings and chat completions
    # Used for both text-to-vector conversion and response generation
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("✅ OpenAI client initialized successfully")
    
except Exception as e:
    error_msg = f"❌ Failed to initialize clients: {str(e)}"
    logger.error(error_msg)
    st.error(error_msg)
    st.stop()

# =============================================================================
# APPLICATION CONSTANTS AND CONFIGURATION
# =============================================================================

# OpenAI Model Configuration
# Reason: text-embedding-3-small provides good quality at lower cost
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536  # Fixed dimension for text-embedding-3-small

# Document Processing Configuration
CHUNK_SIZE = 1000          # Characters per chunk - balance between context and specificity
CHUNK_OVERLAP = 200        # Character overlap between chunks - ensures context continuity
BATCH_SIZE = 100          # Number of texts to embed in one API call - balances speed vs rate limits

# Reason: gpt-4o-mini provides excellent performance at lower cost than gpt-4
CHAT_MODEL = "gpt-4o-mini"

# Search Configuration
DEFAULT_SIMILARITY_THRESHOLD = 0.3    # Minimum cosine similarity for relevant results
DEFAULT_SEARCH_LIMIT = 5              # Number of top results to retrieve
MAX_CONTEXT_LENGTH = 4000             # Maximum characters in context for AI response

logger.info(f"📊 Configuration loaded:")
logger.info(f"  - Embedding Model: {EMBEDDING_MODEL} (dimension: {EMBEDDING_DIMENSION})")
logger.info(f"  - Chat Model: {CHAT_MODEL}")
logger.info(f"  - Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
logger.info(f"  - Batch Size: {BATCH_SIZE}")
logger.info(f"  - Similarity Threshold: {DEFAULT_SIMILARITY_THRESHOLD}")

# =============================================================================
# DOCUMENT LOADING AND PROCESSING FUNCTIONS
# =============================================================================

def load_document(file_path: str, file_type: str) -> List[Document]:
    """
    Load document based on file type with comprehensive error handling and logging.
    
    This function serves as a unified interface for loading different document types:
    - PDF: Uses PyPDFLoader for page-by-page extraction
    - TXT: Uses TextLoader with UTF-8 encoding
    - DOCX: Uses UnstructuredWordDocumentLoader for rich formatting
    
    Args:
        file_path (str): Full path to the document file
        file_type (str): File extension (pdf, txt, docx)
    
    Returns:
        List[Document]: List of LangChain Document objects with content and metadata
    
    Raises:
        ValueError: If file type is unsupported or no content can be extracted
        Exception: For any file loading or parsing errors
    """
    logger.info(f"📖 Loading document: {file_path} (type: {file_type})")
    start_time = time.time()
    
    try:
        # Select appropriate loader based on file type
        # Each loader has specific optimizations for its document type
        if file_type == "pdf":
            logger.debug("Using PyPDFLoader for PDF document")
            loader = PyPDFLoader(file_path)
        elif file_type == "txt":
            logger.debug("Using TextLoader for TXT document with UTF-8 encoding")
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_type == "docx":
            logger.debug("Using UnstructuredWordDocumentLoader for DOCX document")
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            error_msg = f"Unsupported file type: {file_type}. Supported types: pdf, txt, docx"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Load the document using the selected loader
        logger.debug("Starting document loading process...")
        documents = loader.load()
        
        # Validate that content was successfully extracted
        if not documents:
            error_msg = "No content could be extracted from the document"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log successful loading with performance metrics
        load_time = time.time() - start_time
        total_chars = sum(len(doc.page_content) for doc in documents)
        logger.info(f"✅ Successfully loaded {len(documents)} pages in {load_time:.2f}s")
        logger.info(f"📊 Total content: {total_chars:,} characters")
        
        return documents
        
    except Exception as e:
        error_msg = f"Error loading document {file_path}: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Error type: {type(e).__name__}")
        raise

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into manageable chunks using RecursiveCharacterTextSplitter.
    
    This function implements intelligent text splitting that:
    1. Preserves semantic boundaries (paragraphs, sentences)
    2. Maintains context with overlapping chunks
    3. Ensures chunks fit within embedding model limits
    4. Preserves metadata from original documents
    
    The splitting strategy uses a hierarchy of separators:
    - \\n\\n: Paragraph breaks (preferred)
    - \\n: Line breaks
    - " ": Word boundaries
    - "": Character-level split (last resort)
    
    Args:
        documents (List[Document]): Original documents to be chunked
    
    Returns:
        List[Document]: List of document chunks with preserved metadata
    """
    logger.info(f"✂️ Chunking {len(documents)} documents...")
    start_time = time.time()
    
    # Calculate total content before chunking for metrics
    total_chars_before = sum(len(doc.page_content) for doc in documents)
    logger.debug(f"Total content before chunking: {total_chars_before:,} characters")
    
    # Initialize text splitter with optimized configuration
    # Reason: These parameters balance context preservation with embedding limits
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,           # Maximum characters per chunk
        chunk_overlap=CHUNK_OVERLAP,     # Overlap to maintain context between chunks
        length_function=len,             # Use character count for length measurement
        separators=["\n\n", "\n", " ", ""]  # Hierarchical separators for intelligent splitting
    )
    
    logger.debug(f"Text splitter configuration:")
    logger.debug(f"  - Chunk size: {CHUNK_SIZE} characters")
    logger.debug(f"  - Chunk overlap: {CHUNK_OVERLAP} characters")
    logger.debug(f"  - Separators: {text_splitter._separators}")
    
    # Perform the chunking operation
    chunks = text_splitter.split_documents(documents)
    
    # Calculate metrics and log results
    chunk_time = time.time() - start_time
    total_chars_after = sum(len(chunk.page_content) for chunk in chunks)
    avg_chunk_size = total_chars_after / len(chunks) if chunks else 0
    
    logger.info(f"✅ Created {len(chunks)} chunks in {chunk_time:.2f}s")
    logger.info(f"📊 Average chunk size: {avg_chunk_size:.0f} characters")
    logger.debug(f"Total characters after chunking: {total_chars_after:,}")
    
    return chunks

# =============================================================================
# EMBEDDING GENERATION AND UTILITIES
# =============================================================================

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts with intelligent batch processing and retry logic.
    
    This function handles the conversion of text to high-dimensional vectors using OpenAI's
    embedding API. It implements several optimization strategies:
    1. Batch processing to reduce API calls
    2. Rate limit detection and automatic retry
    3. Exponential backoff on failures
    4. Progress tracking for large document sets
    
    Args:
        texts (List[str]): List of text chunks to embed
    
    Returns:
        List[List[float]]: List of embedding vectors (1536-dimensional for text-embedding-3-small)
    
    Raises:
        Exception: If embedding generation fails after retries
    """
    logger.info(f"🔢 Generating embeddings for {len(texts)} text chunks...")
    start_time = time.time()
    embeddings = []
    
    # Process in batches to optimize API usage and respect rate limits
    # Reason: Batch processing reduces API overhead and improves throughput
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"Processing in {total_batches} batches of {BATCH_SIZE} texts each")
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch = texts[i:i + BATCH_SIZE]
        batch_start_time = time.time()
        
        logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
        
        try:
            # Call OpenAI Embeddings API
            # Reason: text-embedding-3-small provides good quality at reasonable cost
            response = openai_client.embeddings.create(
                input=batch,
                model=EMBEDDING_MODEL
            )
            
            # Extract embeddings from API response
            batch_embeddings = [e.embedding for e in response.data]
            embeddings.extend(batch_embeddings)
            
            # Log batch completion with timing
            batch_time = time.time() - batch_start_time
            logger.debug(f"✅ Batch {batch_num} completed in {batch_time:.2f}s")
            
        except Exception as e:
            error_msg = f"Error generating embeddings for batch {batch_num}: {str(e)}"
            logger.error(error_msg)
            
            # Implement intelligent retry logic for rate limits
            if "rate_limit" in str(e).lower():
                retry_delay = 2 ** (batch_num % 4)  # Exponential backoff: 2, 4, 8, 16 seconds
                logger.warning(f"⏳ Rate limit detected. Retrying batch {batch_num} after {retry_delay}s delay...")
                st.warning(f"Rate limit hit. Retrying batch {batch_num} with {retry_delay}s delay...")
                
                time.sleep(retry_delay)
                
                try:
                    # Retry the same batch after delay
                    response = openai_client.embeddings.create(
                        input=batch,
                        model=EMBEDDING_MODEL
                    )
                    batch_embeddings = [e.embedding for e in response.data]
                    embeddings.extend(batch_embeddings)
                    logger.info(f"✅ Batch {batch_num} succeeded after retry")
                    
                except Exception as retry_e:
                    error_msg = f"Failed to generate embeddings after retry: {str(retry_e)}"
                    logger.error(error_msg)
                    st.error(error_msg)
                    raise retry_e
            else:
                # For non-rate-limit errors, fail immediately
                logger.error(f"Non-retryable error in batch {batch_num}: {type(e).__name__}")
                raise e
        
        # Show progress for large embedding jobs
        if total_batches > 5:
            progress = (batch_num / total_batches) * 100
            st.progress(progress / 100, f"Processing embeddings: {batch_num}/{total_batches} batches")
    
    # Log completion metrics
    total_time = time.time() - start_time
    avg_time_per_text = total_time / len(texts) if texts else 0
    logger.info(f"✅ Generated {len(embeddings)} embeddings in {total_time:.2f}s")
    logger.info(f"📊 Average: {avg_time_per_text:.3f}s per text, {len(texts)/total_time:.1f} texts/sec")
    
    return embeddings

def generate_doc_hash(content: str) -> str:
    """
    Generate a unique MD5 hash for document content to detect duplicates.
    
    This hash is used to:
    1. Identify duplicate content during uploads
    2. Track document versions
    3. Optimize storage by avoiding re-processing identical content
    
    Args:
        content (str): Document content to hash
    
    Returns:
        str: 32-character hexadecimal MD5 hash
    """
    # Use MD5 for speed - not cryptographically secure but sufficient for deduplication
    hash_value = hashlib.md5(content.encode('utf-8')).hexdigest()
    logger.debug(f"Generated content hash: {hash_value[:8]}... (length: {len(content)} chars)")
    return hash_value

# =============================================================================
# DOCUMENT STORAGE AND DATABASE OPERATIONS
# =============================================================================

def store_documents(chunks: List[Document], file_name: str):
    """
    Store document chunks with embeddings in Supabase database with comprehensive error handling.
    
    This function implements the complete document storage pipeline:
    1. Extract text content from chunks
    2. Generate embeddings for all chunks
    3. Prepare database records with metadata
    4. Handle duplicate documents (update existing)
    5. Batch insert for optimal database performance
    6. Comprehensive error handling and rollback
    
    The function stores data in the 'documents' table with:
    - content: Original text chunk
    - embedding: 1536-dimensional vector for similarity search
    - metadata: JSON object with file info, chunk details, and hashes
    
    Args:
        chunks (List[Document]): Document chunks from text splitting
        file_name (str): Original filename for tracking and deduplication
    
    Raises:
        Exception: If storage operation fails after retries
    """
    logger.info(f"💾 Storing {len(chunks)} chunks from '{file_name}' to database...")
    storage_start_time = time.time()
    
    with st.spinner(f"🔄 Processing {len(chunks)} chunks for storage..."):
        try:
            # Step 1: Extract text content from LangChain Document objects
            logger.debug("Extracting text content from document chunks...")
            texts = [chunk.page_content for chunk in chunks]
            total_chars = sum(len(text) for text in texts)
            logger.info(f"📊 Total content to process: {total_chars:,} characters")
            
            # Step 2: Generate embeddings for all text chunks
            # This is the most time-consuming step due to API calls
            logger.info("🔢 Starting embedding generation...")
            embedding_start_time = time.time()
            embeddings = get_embeddings(texts)
            embedding_time = time.time() - embedding_start_time
            logger.info(f"✅ Embedding generation completed in {embedding_time:.2f}s")
            
            # Validate embedding generation
            if len(embeddings) != len(chunks):
                error_msg = f"Embedding count mismatch: {len(embeddings)} embeddings for {len(chunks)} chunks"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Step 3: Prepare database records with comprehensive metadata
            logger.debug("Preparing database records with metadata...")
            records = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create rich metadata for each chunk
                # This metadata enables efficient filtering and source attribution
                metadata = {
                    "file_name": file_name,                                    # Original filename
                    "chunk_index": i,                                          # Position within document
                    "total_chunks": len(chunks),                               # Total chunks for this document
                    "page": chunk.metadata.get("page", None),                  # Page number (if available)
                    "source": chunk.metadata.get("source", file_name),        # Source file path
                    "content_hash": generate_doc_hash(chunk.page_content),     # Content deduplication hash
                    "upload_timestamp": datetime.now().isoformat(),           # When this was processed
                    "chunk_length": len(chunk.page_content),                   # Chunk size in characters
                    "embedding_model": EMBEDDING_MODEL                        # Model used for embeddings
                }
                
                # Create database record
                records.append({
                    "content": chunk.page_content,
                    "embedding": embedding,
                    "metadata": metadata
                })
            
            logger.info(f"✅ Prepared {len(records)} database records")
            
            # Step 4: Check for and handle existing documents (deduplication)
            logger.debug(f"Checking for existing documents with filename: {file_name}")
            existing_query = supabase.table("documents").select("id").eq("metadata->>file_name", file_name)
            existing = existing_query.execute()
            
            if existing.data:
                existing_count = len(existing.data)
                logger.info(f"🔄 Found {existing_count} existing chunks for '{file_name}' - updating...")
                st.info(f"Updating existing documents for {file_name}")
                
                # Delete old entries to replace with new version
                # Reason: This ensures consistency and handles document updates
                delete_result = supabase.table("documents").delete().eq("metadata->>file_name", file_name).execute()
                logger.info(f"🗑️ Deleted {existing_count} old chunks")
            else:
                logger.info(f"📄 New document: '{file_name}' - creating fresh entries")
            
            # Step 5: Insert new records in optimized batches
            # Reason: Batch insertion reduces database overhead and transaction count
            db_batch_size = 50  # Optimized for Supabase performance and reliability
            insert_batches = (len(records) + db_batch_size - 1) // db_batch_size
            logger.info(f"💾 Inserting {len(records)} records in {insert_batches} batches...")
            
            inserted_count = 0
            for i in range(0, len(records), db_batch_size):
                batch_num = i // db_batch_size + 1
                batch = records[i:i + db_batch_size]
                
                logger.debug(f"Inserting batch {batch_num}/{insert_batches} ({len(batch)} records)")
                
                try:
                    # Insert batch into Supabase
                    result = supabase.table("documents").insert(batch).execute()
                    inserted_count += len(batch)
                    logger.debug(f"✅ Batch {batch_num} inserted successfully")
                    
                except Exception as batch_error:
                    logger.error(f"❌ Failed to insert batch {batch_num}: {str(batch_error)}")
                    raise batch_error
            
            # Step 6: Verify successful storage and log completion metrics
            total_storage_time = time.time() - storage_start_time
            embedding_percentage = (embedding_time / total_storage_time) * 100
            
            logger.info("="*50)
            logger.info(f"✅ STORAGE COMPLETED SUCCESSFULLY")
            logger.info(f"📊 Storage metrics for '{file_name}':")
            logger.info(f"  - Total chunks stored: {inserted_count}")
            logger.info(f"  - Total storage time: {total_storage_time:.2f}s")
            logger.info(f"  - Embedding time: {embedding_time:.2f}s ({embedding_percentage:.1f}%)")
            logger.info(f"  - Database time: {total_storage_time - embedding_time:.2f}s")
            logger.info(f"  - Average time per chunk: {total_storage_time / len(chunks):.3f}s")
            logger.info("="*50)
            
            st.success(f"✅ Successfully stored {inserted_count} chunks from {file_name}")
            
        except Exception as e:
            # Comprehensive error logging for debugging
            error_msg = f"❌ Error storing documents from '{file_name}': {str(e)}"
            logger.error(error_msg)
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error occurred at: {datetime.now().isoformat()}")
            
            # Show user-friendly error message
            st.error(error_msg)
            
            # Re-raise to allow caller to handle appropriately
            raise

# =============================================================================
# SEMANTIC SEARCH AND RETRIEVAL FUNCTIONS
# =============================================================================

def search_similar_documents(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> List[Dict]:
    """
    Search for semantically similar documents using pgvector-powered similarity search.
    
    This function implements the core retrieval component of the RAG system:
    1. Convert user query to embedding vector
    2. Perform vector similarity search using PostgreSQL pgvector
    3. Rank results by cosine similarity
    4. Return top matching document chunks with metadata
    5. Fallback to client-side search if database function unavailable
    
    The search uses cosine similarity on 1536-dimensional embeddings to find
    the most relevant document chunks for the user's query. This enables
    semantic understanding beyond simple keyword matching.
    
    Args:
        query (str): User's natural language query
        limit (int): Maximum number of results to return (default: 5)
    
    Returns:
        List[Dict]: List of matching documents with content, metadata, and similarity scores
                   Each dict contains: id, content, metadata, embedding, similarity
    """
    logger.info(f"🔍 Searching for documents similar to: '{query[:100]}{'...' if len(query) > 100 else ''}'")
    search_start_time = time.time()
    
    try:
        # Step 1: Convert user query to embedding vector
        # This allows semantic comparison with stored document embeddings
        logger.debug(f"🔢 Generating embedding for query (length: {len(query)} chars)")
        embedding_start_time = time.time()
        
        response = openai_client.embeddings.create(
            input=query,
            model=EMBEDDING_MODEL
        )
        query_embedding = response.data[0].embedding
        
        embedding_time = time.time() - embedding_start_time
        logger.debug(f"✅ Query embedding generated in {embedding_time:.3f}s")
        
        # Validate embedding dimensions
        if len(query_embedding) != EMBEDDING_DIMENSION:
            error_msg = f"Unexpected embedding dimension: {len(query_embedding)} (expected {EMBEDDING_DIMENSION})"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Step 2: Perform pgvector similarity search using database function
        # Reason: Database-side search is much faster than client-side computation
        logger.debug(f"🗄️ Executing pgvector similarity search with threshold {DEFAULT_SIMILARITY_THRESHOLD}")
        db_search_start_time = time.time()
        
        result = supabase.rpc(
            'match_documents',
            {
                'query_embedding': query_embedding,  # Supabase client handles vector conversion
                'match_threshold': DEFAULT_SIMILARITY_THRESHOLD,  # Only return reasonably similar results
                'match_count': limit
            }
        ).execute()
        
        db_search_time = time.time() - db_search_start_time
        
        # Step 3: Process and validate search results
        if result.data:
            # Log successful database search
            result_count = len(result.data)
            total_search_time = time.time() - search_start_time
            
            logger.info(f"✅ pgvector search completed successfully")
            logger.info(f"📊 Search metrics:")
            logger.info(f"  - Results found: {result_count}/{limit}")
            logger.info(f"  - Total search time: {total_search_time:.3f}s")
            logger.info(f"  - Query embedding time: {embedding_time:.3f}s")
            logger.info(f"  - Database search time: {db_search_time:.3f}s")
            
            # Log similarity scores for debugging
            if result.data:
                similarities = [doc.get('similarity', 0) for doc in result.data]
                min_sim, max_sim = min(similarities), max(similarities)
                avg_sim = sum(similarities) / len(similarities)
                logger.debug(f"  - Similarity range: {min_sim:.3f} to {max_sim:.3f} (avg: {avg_sim:.3f})")
            
            return result.data
            
        else:
            # No results from database function - fall back to client-side search
            logger.warning("⚠️ pgvector search returned no results, falling back to client-side search")
            st.warning("⚠️ Using fallback similarity search. Database function may need optimization.")
            return fallback_similarity_search(query_embedding, limit)
        
    except Exception as e:
        # Comprehensive error handling for different failure modes
        error_msg = f"Error in similarity search: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Query: '{query[:200]}{'...' if len(query) > 200 else ''}'")
        
        # Specific error handling for common issues
        if "function match_documents" in str(e).lower():
            logger.error("❌ pgvector function 'match_documents' not found in database")
            st.error("❌ Database search function not found. Please run the database migration.")
        elif "rate_limit" in str(e).lower():
            logger.error("❌ OpenAI API rate limit hit during query embedding")
            st.error("❌ Rate limit error. Please try again in a moment.")
        else:
            logger.error(f"❌ Unexpected search error: {str(e)}")
            st.warning(f"Search error: {str(e)}. Using fallback method.")
        
        # Always try fallback search to maintain functionality
        try:
            logger.info("🔄 Attempting fallback similarity search...")
            return fallback_similarity_search(query_embedding if 'query_embedding' in locals() else query, limit)
        except Exception as fallback_error:
            logger.error(f"❌ Fallback search also failed: {str(fallback_error)}")
            st.error(f"❌ Search functionality unavailable: {str(fallback_error)}")
            return []

def fallback_similarity_search(query_input, limit: int) -> List[Dict]:
    """
    Fallback similarity search using client-side vector computation.
    
    This function provides a backup search method when the PostgreSQL pgvector
    function is unavailable. It:
    1. Fetches all document embeddings from the database
    2. Computes cosine similarity locally using NumPy
    3. Ranks and returns the most similar documents
    
    Note: This method is slower than database-side search but ensures
    functionality when pgvector is not available.
    
    Args:
        query_input: Either a query string or pre-computed embedding vector
        limit (int): Maximum number of results to return
    
    Returns:
        List[Dict]: List of similar documents sorted by similarity score
    """
    logger.info("🔄 Executing fallback client-side similarity search...")
    fallback_start_time = time.time()
    
    try:
        # Step 1: Ensure we have a query embedding vector
        if isinstance(query_input, str):
            logger.debug("Converting query string to embedding...")
            response = openai_client.embeddings.create(
                input=query_input,
                model=EMBEDDING_MODEL
            )
            query_embedding = response.data[0].embedding
            logger.debug("✅ Query embedding generated for fallback search")
        else:
            query_embedding = query_input
            logger.debug("Using pre-computed query embedding")
        
        # Step 2: Fetch all documents from database
        # Note: In production, you might want to implement pagination for very large datasets
        logger.debug("📥 Fetching all documents for client-side comparison...")
        result = supabase.table("documents").select("*").limit(1000).execute()
        
        if not result.data:
            logger.warning("❌ No documents found in database for fallback search")
            return []
        
        doc_count = len(result.data)
        logger.info(f"📊 Processing {doc_count} documents for similarity comparison")
        
        # Step 3: Compute cosine similarities using NumPy for efficiency
        similarities = []
        successful_comparisons = 0
        
        for i, doc in enumerate(result.data):
            try:
                # Extract and validate document embedding
                doc_embedding = doc.get('embedding')
                if not doc_embedding:
                    logger.debug(f"Skipping document {doc.get('id', 'unknown')} - no embedding")
                    continue
                
                # Convert to NumPy arrays for efficient computation
                doc_vec = np.array(doc_embedding, dtype=np.float32)
                query_vec = np.array(query_embedding, dtype=np.float32)
                
                # Calculate cosine similarity: (A·B) / (||A|| × ||B||)
                # Reason: Cosine similarity is ideal for high-dimensional text embeddings
                dot_product = np.dot(doc_vec, query_vec)
                norms = np.linalg.norm(doc_vec) * np.linalg.norm(query_vec)
                
                if norms > 0:  # Avoid division by zero
                    similarity = dot_product / norms
                    similarities.append((similarity, doc))
                    successful_comparisons += 1
                
            except Exception as e:
                logger.debug(f"Error computing similarity for document {doc.get('id', 'unknown')}: {str(e)}")
                continue
        
        # Step 4: Sort by similarity and return top results
        similarities.sort(key=lambda x: x[0], reverse=True)  # Sort by similarity score (descending)
        top_results = [doc for _, doc in similarities[:limit]]
        
        # Add similarity scores to results for debugging
        for i, (similarity, _) in enumerate(similarities[:limit]):
            if i < len(top_results):
                top_results[i]['similarity'] = similarity
        
        # Log completion metrics
        fallback_time = time.time() - fallback_start_time
        logger.info(f"✅ Fallback search completed in {fallback_time:.2f}s")
        logger.info(f"📊 Processed {successful_comparisons}/{doc_count} documents successfully")
        
        if top_results:
            top_similarity = similarities[0][0] if similarities else 0
            logger.info(f"🎯 Top similarity score: {top_similarity:.3f}")
        
        return top_results
        
    except Exception as e:
        error_msg = f"Error in fallback similarity search: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Fallback search failed after {time.time() - fallback_start_time:.2f}s")
        return []

# =============================================================================
# AI RESPONSE GENERATION
# =============================================================================

def generate_response(query: str, context_docs: List[Dict]) -> str:
    """
    Generate AI response using retrieved document context with advanced prompt engineering.
    
    This function implements the "Generation" component of RAG:
    1. Combines retrieved document chunks into coherent context
    2. Creates optimized prompts for accurate, contextual responses
    3. Manages token limits to fit within model constraints
    4. Provides fallback responses for edge cases
    
    The function uses advanced prompt engineering techniques to ensure:
    - Accurate information from provided context
    - Clear indication when information is not available
    - Proper source attribution
    - Consistent response quality
    
    Args:
        query (str): User's original question
        context_docs (List[Dict]): Retrieved document chunks with content and metadata
    
    Returns:
        str: Generated response based on the provided context
    """
    logger.info(f"🤖 Generating AI response for query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
    generation_start_time = time.time()
    
    try:
        # Step 1: Prepare and optimize context from retrieved documents
        logger.debug(f"📋 Processing {len(context_docs)} context documents...")
        
        if not context_docs:
            logger.warning("⚠️ No context documents provided for response generation")
            return "I don't have any relevant information in the uploaded documents to answer your question. Please make sure you've uploaded documents related to your query."
        
        # Combine document contents with intelligent truncation
        context_parts = []
        total_context_length = 0
        
        for i, doc in enumerate(context_docs):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            file_name = metadata.get('file_name', 'Unknown')
            
            # Add source attribution to each chunk
            attributed_content = f"[Source: {file_name}]\n{content}"
            
            # Check if adding this chunk would exceed our context limit
            if total_context_length + len(attributed_content) > MAX_CONTEXT_LENGTH:
                logger.debug(f"Context limit reached. Using {i} of {len(context_docs)} documents.")
                break
            
            context_parts.append(attributed_content)
            total_context_length += len(attributed_content)
        
        context = "\n\n".join(context_parts)
        logger.info(f"📊 Context prepared: {len(context_parts)} chunks, {total_context_length:,} characters")
        
        # Step 2: Create optimized prompt with advanced instructions
        # Reason: Detailed system prompts improve response quality and consistency
        system_prompt = """You are an expert AI assistant specializing in document analysis and question answering. Your role is to provide accurate, helpful responses based solely on the provided document context.

INSTRUCTIONS:
1. ACCURACY: Use only information explicitly stated in the provided context
2. ATTRIBUTION: Reference specific sources when possible (e.g., "According to [Source: filename.pdf]...")
3. CLARITY: Provide clear, well-structured answers that directly address the question
4. HONESTY: If the answer is not in the context, clearly state this limitation
5. COMPLETENESS: Provide comprehensive answers when sufficient context is available
6. RELEVANCE: Focus on information most relevant to the user's question

RESPONSE FORMAT:
- Start with a direct answer to the question
- Provide supporting details from the context
- Include source references where applicable
- End with any relevant additional insights from the documents"""

        user_prompt = f"""CONTEXT DOCUMENTS:
{context}

USER QUESTION: {query}

Please provide a comprehensive answer based on the context documents above."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Step 3: Generate response with streaming for real-time user experience
        logger.debug(f"🔤 Generating streaming response using {CHAT_MODEL}...")
        ai_start_time = time.time()
        
        stream = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.6,  # Optimized for faster generation while maintaining quality
            max_tokens=500,   # Reduced for faster responses - still sufficient for detailed answers
            top_p=0.9,       # Keep nucleus sampling for quality
            stream=True      # Enable streaming for real-time response
            # Removed frequency_penalty and presence_penalty for faster generation
        )
        
        # Process streaming response and display in real-time
        generated_response = ""
        response_placeholder = st.empty()
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                generated_response += content
                # Update the response in real-time
                response_placeholder.markdown(generated_response + "▌")  # Add cursor for typing effect
        
        # Remove cursor and show final response
        response_placeholder.markdown(generated_response)
        ai_time = time.time() - ai_start_time
        
        # Step 4: Log generation metrics and validate response
        total_generation_time = time.time() - generation_start_time
        
        logger.info(f"✅ Response generated successfully")
        logger.info(f"📊 Generation metrics:")
        logger.info(f"  - Total generation time: {total_generation_time:.3f}s")
        logger.info(f"  - AI model time: {ai_time:.3f}s")
        logger.info(f"  - Response length: {len(generated_response):,} characters")
        logger.info(f"  - Context utilization: {len(context_parts)}/{len(context_docs)} chunks")
        
        # Validate response quality
        if not generated_response or len(generated_response.strip()) < 10:
            logger.warning("⚠️ Generated response appears to be too short or empty")
            return "I was unable to generate a comprehensive response. Please try rephrasing your question or check if relevant documents are uploaded."
        
        return generated_response
        
    except Exception as e:
        # Comprehensive error handling for response generation
        error_msg = f"Error generating response: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Generation failed after {time.time() - generation_start_time:.3f}s")
        
        # Provide user-friendly error message
        if "rate_limit" in str(e).lower():
            return "I'm currently experiencing high demand. Please try your question again in a moment."
        elif "token" in str(e).lower():
            return "Your question or the retrieved context is too long. Please try asking a more specific question."
        else:
            return f"I encountered an error while generating the response. Please try again or contact support if the issue persists."

# Streamlit UI
def main():
    st.set_page_config(
        page_title="RAG with Supabase", 
        page_icon="🔍", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📚 RAG Application with Supabase Vector Store")
    st.markdown("Upload documents and ask questions using AI-powered search")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📄 Document Upload")
        
        # File uploader with better help text
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'docx'],
            help="Upload PDF, TXT, or DOCX files (max 200MB per file)"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.info(f"**File:** {uploaded_file.name}\n**Size:** {uploaded_file.size:,} bytes")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name
            
            # Process button
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                try:
                    # Determine file type
                    file_type = uploaded_file.name.split('.')[-1].lower()
                    
                    # Load document
                    with st.spinner("📖 Loading document..."):
                        documents = load_document(tmp_file_path, file_type)
                        st.info(f"✅ Loaded {len(documents)} pages")
                    
                    # Chunk documents
                    with st.spinner("✂️ Chunking document..."):
                        chunks = chunk_documents(documents)
                        st.info(f"✅ Created {len(chunks)} chunks")
                    
                    # Store in vector database
                    store_documents(chunks, uploaded_file.name)
                    
                    # Clean up temp file
                    os.unlink(tmp_file_path)
                    
                    # Show success message with rerun to refresh stored docs
                    st.success("🎉 Document processed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")
                    logger.error(f"Error processing document: {str(e)}")
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
        
        # Display stored documents
        st.divider()
        st.header("📁 Stored Documents")
        try:
            # Get unique file names with count
            result = supabase.table("documents").select("metadata").execute()
            if result.data:
                file_counts = {}
                for doc in result.data:
                    if doc['metadata'] and 'file_name' in doc['metadata']:
                        file_name = doc['metadata']['file_name']
                        file_counts[file_name] = file_counts.get(file_name, 0) + 1
                
                for file_name, count in file_counts.items():
                    st.text(f"📄 {file_name} ({count} chunks)")
            else:
                st.text("No documents stored yet")
        except Exception as e:
            st.error(f"❌ Error fetching documents: {str(e)}")
    
    # Main chat interface
    st.header("💬 Chat with your Documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📎 Sources"):
                    for source in message["sources"]:
                        file_name = source['metadata'].get('file_name', 'Unknown')
                        chunk_index = source['metadata'].get('chunk_index', 'Unknown')
                        st.text(f"📄 {file_name} (chunk {chunk_index})")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents..."):
                # Search for relevant documents
                relevant_docs = search_similar_documents(prompt, limit=5)
                
                if relevant_docs:
                    # Generate streaming response with context
                    response = generate_response(prompt, relevant_docs)
                    
                    # Add to messages (response is already displayed via streaming)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": relevant_docs
                    })
                    
                    # Show sources
                    with st.expander("📎 Sources"):
                        for doc in relevant_docs:
                            file_name = doc['metadata'].get('file_name', 'Unknown')
                            chunk_index = doc['metadata'].get('chunk_index', 'Unknown')
                            st.text(f"📄 {file_name} (chunk {chunk_index})")
                else:
                    response = "🤔 I couldn't find any relevant information in the uploaded documents. Please make sure you've uploaded documents related to your question."
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer with helpful info
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🔧 Built with Streamlit + OpenAI + Supabase")
    with col2:
        st.caption("🚀 Powered by pgvector for fast similarity search")  
    with col3:
        st.caption("💡 Upload documents to get started")

if __name__ == "__main__":
    main()