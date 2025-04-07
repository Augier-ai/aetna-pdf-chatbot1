"""
LLM service for interacting with OpenAI models.
"""
import logging
from typing import List, Dict, Any, Optional
from langchain.schema.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from src.config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMService:
    """
    Handles interactions with OpenAI models for generating responses.
    """
    
    def __init__(self, vector_store):
        """
        Initialize the LLM service.
        
        Args:
            vector_store: Vector store for retrieval
        """
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.qa_chain = self._create_qa_chain()
    
    def _create_qa_chain(self) -> ConversationalRetrievalChain:
        """
        Create a conversational retrieval chain.
        
        Returns:
            ConversationalRetrievalChain
        """
        # Create custom prompt templates
        condense_question_template = """
        Given the following conversation and a follow up question, rephrase the follow up question 
        to be a standalone question that captures all relevant context from the conversation.
        
        Chat History:
        {chat_history}
        
        Follow Up Input: {question}
        Standalone question:
        """
        
        condense_question_prompt = PromptTemplate.from_template(condense_question_template)
        
        qa_template = """
        You are an AI assistant for Aetna Better Health Illinois. You provide helpful, accurate, and friendly information 
        about the Aetna Better Health Illinois Member Handbook. Your goal is to assist members in understanding their 
        benefits, coverage, and how to navigate their healthcare.
        
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say 
        that you don't know, don't try to make up an answer. Always maintain a helpful and professional tone.
        
        {context}
        
        Question: {question}
        
        Helpful Answer:
        """
        
        qa_prompt = PromptTemplate.from_template(qa_template)
        
        # Create the chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            condense_question_prompt=condense_question_prompt,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )
        
        return qa_chain
    
    def get_response(self, query: str) -> str:
        """
        Get a response from the LLM for a given query.
        
        Args:
            query: User query
            
        Returns:
            LLM response
        """
        logger.info(f"Getting response for query: {query}")
        
        try:
            response = self.qa_chain({"question": query})
            return response["answer"]
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            return f"I'm sorry, I encountered an error while processing your request. Please try again later. Error: {str(e)}"
    
    def reset_conversation(self) -> None:
        """
        Reset the conversation history.
        """
        logger.info("Resetting conversation history")
        self.memory.clear()
