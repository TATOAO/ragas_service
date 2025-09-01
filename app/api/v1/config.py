import logging

from fastapi import APIRouter, Depends

from app.core.auth import get_current_user_api_key
from app.models import EmbeddingProvidersResponse, LLMProvidersResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/llm-providers", response_model=LLMProvidersResponse)
async def list_llm_providers(
    current_user: dict = Depends(get_current_user_api_key)
):
    """List available LLM providers and models"""
    providers = [
        {
            "name": "openai",
            "models": [
                {
                    "name": "gpt-4o",
                    "max_tokens": 4096,
                    "cost_per_1k_tokens": 0.03
                },
                {
                    "name": "gpt-4o-mini",
                    "max_tokens": 4096,
                    "cost_per_1k_tokens": 0.015
                },
                {
                    "name": "gpt-3.5-turbo",
                    "max_tokens": 4096,
                    "cost_per_1k_tokens": 0.002
                }
            ]
        },
        {
            "name": "anthropic",
            "models": [
                {
                    "name": "claude-3-opus",
                    "max_tokens": 4096,
                    "cost_per_1k_tokens": 0.015
                },
                {
                    "name": "claude-3-sonnet",
                    "max_tokens": 4096,
                    "cost_per_1k_tokens": 0.003
                },
                {
                    "name": "claude-3-haiku",
                    "max_tokens": 4096,
                    "cost_per_1k_tokens": 0.00025
                }
            ]
        },
        {
            "name": "google",
            "models": [
                {
                    "name": "gemini-pro",
                    "max_tokens": 8192,
                    "cost_per_1k_tokens": 0.001
                }
            ]
        }
    ]
    
    return {"providers": providers}


@router.get("/embedding-providers", response_model=EmbeddingProvidersResponse)
async def list_embedding_providers(
    current_user: dict = Depends(get_current_user_api_key)
):
    """List available embedding providers and models"""
    providers = [
        {
            "name": "openai",
            "models": [
                {
                    "name": "text-embedding-3-small",
                    "dimensions": 1536,
                    "cost_per_1k_tokens": 0.00002
                },
                {
                    "name": "text-embedding-3-large",
                    "dimensions": 3072,
                    "cost_per_1k_tokens": 0.00013
                }
            ]
        },
        {
            "name": "cohere",
            "models": [
                {
                    "name": "embed-english-v3.0",
                    "dimensions": 1024,
                    "cost_per_1k_tokens": 0.0001
                }
            ]
        },
        {
            "name": "sentence-transformers",
            "models": [
                {
                    "name": "all-MiniLM-L6-v2",
                    "dimensions": 384,
                    "cost_per_1k_tokens": 0.0
                }
            ]
        }
    ]
    
    return {"providers": providers}


# python -m app.api.v1.config
def main():
    """Unit test function for config routes"""
    from fastapi.testclient import TestClient
    from main import app
    
    client = TestClient(app)
    
    # Test list LLM providers
    response = client.get("/api/v1/config/llm-providers", headers={"Authorization": "Bearer test-api-key"})
    print(response.json())
    assert response.status_code == 200
    
    # Test list embedding providers
    response = client.get("/api/v1/config/embedding-providers", headers={"Authorization": "Bearer test-api-key"})
    print(response.json())
    assert response.status_code == 200
    
    print("Config routes test passed!")


if __name__ == "__main__":
    main()
