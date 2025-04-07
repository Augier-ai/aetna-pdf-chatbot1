"""
Vector store operations for storing and retrieving document embeddings.
"""
import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.document import Document

from src.config import OPENAI_API_KEY, VECTOR_DB_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Handles vector database operations for storing and retrieving document embeddings.
    """
    
    def __init__(self, vector_db_path: str = VECTOR_DB_PATH):
        """
        Initialize the vector store.
        
        Args:
            vector_db_path: Path to store the vector database
        """
        self.vector_db_path = vector_db_path
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.vector_store = None
        
        # Create directory if it doesn't exist
        os.makedirs(self.vector_db_path, exist_ok=True)
    
    def _convert_to_documents(self, chunks: List[Dict[str, Any]]) -> List[Document]:
        """
        Convert chunks to LangChain Document objects.
        
        Args:
            chunks: List of dictionaries containing text and metadata
            
        Returns:
            List of Document objects
        """
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"]
            )
            documents.append(doc)
        
        return documents
    
    def create_vector_store(self, chunks: List[Dict[str, Any]]) -> Chroma:
        """
        Create a new vector store from text chunks.
        
        Args:
            chunks: List of dictionaries containing text and metadata
            
        Returns:
            Chroma vector store
        """
        documents = self._convert_to_documents(chunks)
        
        logger.info(f"Creating vector store with {len(documents)} documents")
        
        # Create vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.vector_db_path
        )
        
        # Persist to disk
        vector_store.persist()
        
        self.vector_store = vector_store
        logger.info(f"Vector store created and persisted to {self.vector_db_path}")
        
        return vector_store
    
    def load_vector_store(self) -> Optional[Chroma]:
        """
        Load an existing vector store from disk.
        
        Returns:
            Chroma vector store or None if it doesn't exist
        """
        if not os.path.exists(self.vector_db_path) or not os.listdir(self.vector_db_path):
            logger.warning(f"Vector store not found at {self.vector_db_path}")
            return None
        
        logger.info(f"Loading vector store from {self.vector_db_path}")
        
        vector_store = Chroma(
            persist_directory=self.vector_db_path,
            embedding_function=self.embeddings
        )
        
        self.vector_store = vector_store
        logger.info("Vector store loaded successfully")
        
        return vector_store
    
    def get_or_create_vector_store(self, chunks: Optional[List[Dict[str, Any]]] = None) -> Chroma:
        """
        Get existing vector store or create a new one if it doesn't exist.
        
        Args:
            chunks: List of dictionaries containing text and metadata (required if creating new store)
            
        Returns:
            Chroma vector store
        """
        vector_store = self.load_vector_store()
        
        if vector_store is None:
            if chunks is None:
                raise ValueError("Chunks must be provided when creating a new vector store")
            
            vector_store = self.create_vector_store(chunks)
        
        return vector_store
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform similarity search to find relevant documents.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of relevant Document objects
        """
        if self.vector_store is None:
            self.vector_store = self.load_vector_store()
            if self.vector_store is None:
                raise ValueError("Vector store not found. Please create it first.")
        
        logger.info(f"Performing similarity search for query: {query}")
        
        docs = self.vector_store.similarity_search(query, k=k)
        
        logger.info(f"Found {len(docs)} relevant documents")
        
        return docs
