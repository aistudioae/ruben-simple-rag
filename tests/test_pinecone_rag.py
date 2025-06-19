"""
Unit tests for Pinecone RAG application
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the Streamlit import
sys.modules['streamlit'] = MagicMock()

import pinecone_rag


class TestPineconeRAG:
    """Test suite for Pinecone RAG functions"""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client"""
        client = Mock()
        # Mock embedding response
        embedding_response = Mock()
        embedding_response.data = [Mock(embedding=[0.1] * 1536)]
        client.embeddings.create.return_value = embedding_response
        
        # Mock chat completion response
        chat_response = Mock()
        chat_response.choices = [Mock(message=Mock(content="Test response"))]
        client.chat.completions.create.return_value = chat_response
        
        return client
    
    @pytest.fixture
    def mock_pinecone_index(self):
        """Mock Pinecone index"""
        index = Mock()
        
        # Mock describe_index_stats
        stats = Mock()
        stats.total_vector_count = 100
        stats.dimension = 1536
        index.describe_index_stats.return_value = stats
        
        # Mock query response
        query_response = Mock()
        query_response.matches = [
            Mock(
                id="test_0",
                score=0.95,
                metadata={
                    "content": "Test content",
                    "filename": "test.txt",
                    "chunk_index": 0
                }
            )
        ]
        index.query.return_value = query_response
        
        return index
    
    def test_generate_embeddings(self, mock_openai_client):
        """Test embedding generation"""
        # Test normal operation
        result = pinecone_rag.generate_embeddings("Test text", mock_openai_client)
        assert len(result) == 1536
        assert all(isinstance(x, float) for x in result)
        mock_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="Test text"
        )
    
    def test_semantic_search(self, mock_pinecone_index, mock_openai_client):
        """Test semantic search functionality"""
        # Test search
        results = pinecone_rag.semantic_search(
            "Test query",
            mock_pinecone_index,
            mock_openai_client,
            top_k=5
        )
        
        # Verify results
        assert len(results) == 1
        assert results[0].score == 0.95
        assert results[0].metadata["content"] == "Test content"
        
        # Verify embedding was generated
        mock_openai_client.embeddings.create.assert_called_once()
        
        # Verify index was queried
        mock_pinecone_index.query.assert_called_once()
    
    def test_generate_response(self, mock_openai_client):
        """Test response generation with context"""
        # Mock context
        context = [
            Mock(metadata={"content": "Context 1", "filename": "file1.txt"}),
            Mock(metadata={"content": "Context 2", "filename": "file2.txt"})
        ]
        
        # Generate response
        response = pinecone_rag.generate_response(
            "Test question",
            context,
            mock_openai_client
        )
        
        # Verify response
        assert response == "Test response"
        
        # Verify OpenAI was called
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-4o-mini"
        assert len(call_args[1]["messages"]) == 2
    
    @patch('pinecone_rag.PyPDFLoader')
    @patch('pinecone_rag.RecursiveCharacterTextSplitter')
    @patch('tempfile.NamedTemporaryFile')
    @patch('os.unlink')
    def test_load_document_pdf(self, mock_unlink, mock_tempfile, mock_splitter, mock_loader):
        """Test PDF document loading"""
        # Mock file
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.getvalue.return_value = b"PDF content"
        
        # Mock temp file
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.pdf"
        mock_tempfile.return_value.__enter__.return_value = mock_temp
        
        # Mock loader
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [Mock(page_content="Test content")]
        mock_loader.return_value = mock_loader_instance
        
        # Mock splitter
        mock_splitter_instance = Mock()
        mock_splitter_instance.split_documents.return_value = [
            Mock(page_content="Chunk 1"),
            Mock(page_content="Chunk 2")
        ]
        mock_splitter.return_value = mock_splitter_instance
        
        # Test loading
        chunks = pinecone_rag.load_document(mock_file)
        
        # Verify
        assert len(chunks) == 2
        mock_loader.assert_called_once_with("/tmp/test.pdf")
        mock_unlink.assert_called_once_with("/tmp/test.pdf")
    
    def test_store_in_pinecone(self, mock_pinecone_index, mock_openai_client):
        """Test storing documents in Pinecone"""
        # Mock chunks
        chunks = [
            Mock(page_content="Chunk 1"),
            Mock(page_content="Chunk 2")
        ]
        
        # Store chunks
        pinecone_rag.store_in_pinecone(
            chunks,
            mock_pinecone_index,
            mock_openai_client,
            "test.txt"
        )
        
        # Verify embeddings were generated
        assert mock_openai_client.embeddings.create.call_count == 2
        
        # Verify upsert was called
        mock_pinecone_index.upsert.assert_called_once()
        call_args = mock_pinecone_index.upsert.call_args
        vectors = call_args[1]["vectors"]
        assert len(vectors) == 2
        assert all("id" in v and "values" in v and "metadata" in v for v in vectors)
    
    def test_load_document_unsupported_type(self):
        """Test loading unsupported file type"""
        # Mock file with unsupported extension
        mock_file = Mock()
        mock_file.name = "test.xyz"
        mock_file.getvalue.return_value = b"Content"
        
        # Test that it raises ValueError
        with patch('tempfile.NamedTemporaryFile'), patch('os.unlink'):
            with pytest.raises(ValueError, match="Unsupported file type"):
                pinecone_rag.load_document(mock_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 