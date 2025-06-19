# SimpleRAG - Retrieval-Augmented Generation Application

A powerful RAG application built with Streamlit and OpenAI that allows you to upload documents and ask questions about them using AI-powered search. Includes implementations for multiple vector databases.

## 🚀 Features

- **Multi-format Support**: Upload PDF, TXT, and DOCX files
- **Smart Chunking**: Intelligent document splitting for optimal retrieval
- **Multiple Vector Database Options**:
  - **Supabase** with pgvector (`simpleRAG.py`)
  - **Pinecone** serverless vector database (`pinecone_rag.py`)
- **AI Chat Interface**: Natural language Q&A with context-aware responses
- **Real-time Processing**: Live document processing with progress indicators
- **Source Citations**: See which document chunks were used for each answer

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Vector Database**: Supabase (PostgreSQL + pgvector)
- **AI Models**: OpenAI (GPT-4o-mini for chat, text-embedding-3-small for embeddings)
- **Document Processing**: LangChain (community loaders and text splitters)
- **Language**: Python 3.9+

## 📋 Prerequisites

- Python 3.9 or higher
- OpenAI API key
- **For Supabase version**: Supabase account and project
- **For Pinecone version**: Pinecone account (free tier available)

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd SimpleRAG
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   
   For Supabase version:
   ```bash
   pip install -r requirements.txt
   ```
   
   For Pinecone version:
   ```bash
   pip install -r requirements_pinecone.txt
   ```

4. **Set up environment variables**:
   
   For Supabase version:
   ```bash
   export SUPABASE_URL="your-supabase-url"
   export SUPABASE_KEY="your-supabase-anon-key"
   export OPENAI_API_KEY="your-openai-api-key"
   ```
   
   For Pinecone version:
   ```bash
   cp env.pinecone.template .env.pinecone
   # Edit .env.pinecone with your API keys
   ```

## 🚀 Usage

1. **Start the application**:
   ```bash
   source venv/bin/activate
   streamlit run simpleRAG.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Upload documents** using the sidebar file uploader

4. **Ask questions** about your documents in the chat interface

## 🧪 Testing

Run the test suite to verify everything is working:

```bash
source venv/bin/activate
python test_rag.py
```

Test with the provided sample document:
1. Upload `test_document.txt`
2. Ask questions like:
   - "What is machine learning?"
   - "What are the applications of AI?"
   - "Explain deep learning"

## 📊 Database Schema

The application uses a simple but powerful schema:

```sql
CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  content TEXT,
  embedding VECTOR(1536),
  metadata JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## 🎯 Key Improvements Made

### ✅ Updated Dependencies
- Upgraded to latest compatible versions of all libraries
- Fixed compatibility issues between LangChain and OpenAI
- Updated Supabase client to latest version

### ✅ Enhanced Error Handling
- Better error messages and user feedback
- Graceful fallbacks for API failures
- Retry logic for rate limiting

### ✅ Improved Performance
- Added pgvector function for server-side similarity search
- Batch processing for embeddings and database operations
- Optimized chunking strategy

### ✅ Better User Experience
- Enhanced UI with emojis and better organization
- Progress indicators for long operations
- File information display
- Source citation in chat responses

### ✅ Robust Architecture
- Environment variable validation
- Client initialization with error handling
- Logging for debugging
- Virtual environment isolation

## 🔍 How It Works

1. **Document Upload**: Files are processed using LangChain loaders
2. **Text Chunking**: Documents are split into manageable chunks with overlap
3. **Embedding Generation**: OpenAI creates vector embeddings for each chunk
4. **Vector Storage**: Embeddings are stored in Supabase with metadata
5. **Similarity Search**: User queries are embedded and matched against stored vectors
6. **Response Generation**: Relevant chunks provide context for AI-generated answers

## 🛡️ Security Notes

- API keys are stored as environment variables (not hardcoded)
- Supabase handles database security and authentication
- File uploads are processed in temporary directories
- No sensitive data is logged

## 🐛 Troubleshooting

### Common Issues:

1. **Import Errors**: Ensure virtual environment is activated
2. **API Errors**: Check your OpenAI API key and credits
3. **Database Errors**: Verify Supabase credentials and table exists
4. **Memory Issues**: Large documents may need chunking adjustments

### Debug Mode:
Set logging level to DEBUG in the script for more detailed output.

## 📈 Performance Tips

- Use smaller chunk sizes for better precision
- Increase chunk overlap for better context preservation
- Monitor OpenAI usage to avoid rate limits
- Regular database maintenance for optimal performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- OpenAI for the powerful language models
- Supabase for the excellent vector database
- LangChain for document processing tools
- Streamlit for the amazing web app framework 