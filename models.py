import os
import logging
from typing import Optional, Any
import requests
from pathlib import Path


logger = logging.getLogger(__name__)


def check_ollama_available(model_name: str) -> bool:
    """Check if Ollama model is available locally."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any(m["name"] == model_name for m in models)
        return False
    except Exception:
        return False

def get_openai_apikey(api_key_source: str) -> str:
    """
    Get OpenAI API key from specified source.
    
    Args:
        api_key_source: Source of API key ("env" or "file")
    
    Returns:
        str: API key
        
    Raises:
        ValueError: If API key not found or source invalid
    """
    if api_key_source == "env":
        from dotenv import load_dotenv
        # Load environment variables from .env file
        load_dotenv()
        # Get API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return api_key
        
    elif api_key_source == "file":
        api_key_file = os.getenv('OPENAI_API_KEY_FILE')
        if not api_key_file:
            raise ValueError("OPENAI_API_KEY_FILE not set in environment variables")
            
        try:
            with open(api_key_file) as f:
                return f.read().strip()
        except Exception as e:
            raise ValueError(f"Failed to read API key file: {e}")
            
    else:
        raise ValueError(f"Invalid api_key_source: {api_key_source}. Must be 'env' or 'file'")

def create_model(
    model_name: str,
    model_type: str = "openai",
    api_key_source: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Create a language model instance.
    
    Args:
        model_name: Name of the model to create
        model_type: Type of model ("openai" or "ollama")
        api_key_source: Source of API key for OpenAI models ("env" or "file")
        **kwargs: Additional model parameters
    
    Returns:
        Language model instance
    """
    try:
        if model_type == "openai":
            from langchain_openai import ChatOpenAI
            
            if not api_key_source:
                raise ValueError("api_key_source must be specified for OpenAI models")
                
            # Get API key
            openai_api_key = get_openai_apikey(api_key_source)
            
            return ChatOpenAI(
                model=model_name,
                openai_api_key=openai_api_key,
                **kwargs
            )
            
        elif model_type == "ollama":
            from langchain_ollama import OllamaLLM
            
            # Check if model is available
            if not check_ollama_available(model_name):
                logger.warning(f"Model {model_name} not found. Please pull it using 'ollama pull {model_name}'")
                raise ValueError(f"Ollama model {model_name} not available")
            
            return OllamaLLM(
                model=model_name,
                **kwargs
            )
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test OpenAI model
    try:
        model = create_model("gpt-3.5-turbo", model_type="openai")
        response = model.invoke("Hello!")
        print("OpenAI response:", response)
    except Exception as e:
        print("OpenAI test failed:", e)
    
    # Test Ollama model
    try:
        model = create_model("llama2", model_type="ollama")
        response = model.invoke("Hello!")
        print("Ollama response:", response)
    except Exception as e:
        print("Ollama test failed:", e)
