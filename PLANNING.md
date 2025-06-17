# SimpleRAG Project Planning

## Project Overview
SimpleRAG is a Retrieval-Augmented Generation (RAG) application built with Python that allows users to upload documents and ask questions about their content using AI.

## Architecture
- **Frontend**: Streamlit web interface
- **Backend**: Python with FastAPI-style processing
- **Database**: Supabase with pgvector extension for vector storage
- **AI Services**: OpenAI (embeddings + chat completion)
- **Document Processing**: LangChain for PDF/TXT/DOCX parsing

## Technology Stack
- **UI Framework**: Streamlit ≥1.31.0
- **Database**: Supabase ≥2.8.0 with pgvector
- **AI/ML**: OpenAI ≥1.52.0 (text-embedding-3-small, gpt-4o-mini)
- **Document Processing**: LangChain ≥0.3.0
- **Vector Operations**: pgvector with cosine similarity

## Current Status
✅ **Completed**:
- Core RAG functionality implemented
- Modern dependency updates completed
- Supabase database schema created
- Vector similarity search function `match_documents` deployed
- Streamlit UI with file upload and chat interface
- Document chunking and embedding pipeline
- Deployment configuration for Render

⚠️ **Issues to Debug**:
- Vector similarity search returning empty results despite documents being stored
- RPC calls to `match_documents` function may not be working correctly
- Need to verify embedding dimensions and similarity thresholds

## Deployment
- **Platform**: Render
- **Configuration**: render.yaml with environment variables
- **URL**: Will be provided after deployment

## Next Steps
1. Deploy to Render platform
2. Test vector similarity search in production
3. Debug search functionality if issues persist
4. Monitor performance and costs

## Development Guidelines
- Files should not exceed 500 lines
- All new features need Pytest unit tests
- Code should be well-documented
- Use environment variables for sensitive data 