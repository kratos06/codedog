from functools import lru_cache
from os import environ as env

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI


@lru_cache(maxsize=1)
def load_gpt_llm() -> BaseChatModel:
    """Load GPT 3.5 Model"""
    if env.get("AZURE_OPENAI"):
        llm = AzureChatOpenAI(
            openai_api_type="azure",
            api_key=env.get("AZURE_OPENAI_API_KEY", ""),
            azure_endpoint=env.get("AZURE_OPENAI_API_BASE", ""),
            api_version="2024-05-01-preview",
            azure_deployment=env.get("AZURE_OPENAI_DEPLOYMENT_ID", "gpt-35-turbo"),
            model="gpt-3.5-turbo",
            temperature=0,
        )
    else:
        llm = ChatOpenAI(
            api_key=env.get("OPENAI_API_KEY"),
            model="gpt-3.5-turbo",
        )
    return llm


@lru_cache(maxsize=1)
def load_gpt4_llm():
    """
    Load and return a GPT-4 language model instance.
    
    This function creates a GPT-4 model based on the environment configuration. If the
    'AZURE_OPENAI' variable is set, it initializes an AzureChatOpenAI instance configured
    with Azure-specific parameters; otherwise, it creates a ChatOpenAI instance using the
    OpenAI API. Note that the function does not verify if the provided API key has access
    to GPT-4.
    """
    if env.get("AZURE_OPENAI"):
        llm = AzureChatOpenAI(
            openai_api_type="azure",
            api_key=env.get("AZURE_OPENAI_API_KEY", ""),
            azure_endpoint=env.get("AZURE_OPENAI_API_BASE", ""),
            api_version="2024-05-01-preview",
            azure_deployment=env.get("AZURE_OPENAI_DEPLOYMENT_ID", "gpt-35-turbo"),
            model="gpt-4",
            temperature=0,
        )
    else:
        llm = ChatOpenAI(
            api_key=env.get("OPENAI_API_KEY"),
            model="gpt-4",
        )
    return llm
