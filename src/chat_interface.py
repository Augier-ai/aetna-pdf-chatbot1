"""
Gradio chat interface for the Aetna PDF Chatbot.
"""
import logging
import gradio as gr
from typing import List, Dict, Any, Tuple, Optional

from src.config import APP_TITLE, APP_DESCRIPTION, DEFAULT_GREETING
from src.llm_service import LLMService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatInterface:
    """
    Gradio chat interface for interacting with the Aetna PDF Chatbot.
    """
    
    def __init__(self, llm_service: LLMService):
        """
        Initialize the chat interface.
        
        Args:
            llm_service: LLM service for generating responses
        """
        self.llm_service = llm_service
        self.interface = self._create_interface()
    
    def _create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title=APP_TITLE) as interface:
            # Header
            gr.Markdown(f"# {APP_TITLE}")
            gr.Markdown(APP_DESCRIPTION)
            
            # Chat interface
            chatbot = gr.Chatbot(
                height=500,
                bubble_full_width=False,
                show_copy_button=True,
                avatar_images=("👤", "🏥"),
            )
            
            # Input components
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask a question about your Aetna Better Health Illinois benefits...",
                    scale=9,
                    container=False,
                )
                submit_btn = gr.Button("Send", scale=1, variant="primary")
            
            # Clear button
            clear_btn = gr.Button("Clear Conversation")
            
            # Add system message on page load
            gr.on(
                triggers=[gr.load],
                fn=lambda: [[None, DEFAULT_GREETING]],
                outputs=[chatbot],
            )
            
            # Handle message submission
            def respond(message, chat_history):
                if not message.strip():
                    return chat_history, ""
                
                # Add user message to chat history
                chat_history.append([message, None])
                
                # Get response from LLM
                response = self.llm_service.get_response(message)
                
                # Update chat history with response
                chat_history[-1][1] = response
                
                return chat_history, ""
            
            # Connect event handlers
            submit_btn.click(
                respond,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg],
            )
            
            msg.submit(
                respond,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg],
            )
            
            # Clear conversation
            def clear_conversation():
                self.llm_service.reset_conversation()
                return [[None, DEFAULT_GREETING]]
            
            clear_btn.click(
                clear_conversation,
                outputs=[chatbot],
            )
        
        return interface
    
    def launch(self, **kwargs):
        """
        Launch the Gradio interface.
        
        Args:
            **kwargs: Additional arguments to pass to gr.launch()
        """
        logger.info("Launching Gradio interface")
        self.interface.launch(**kwargs)
