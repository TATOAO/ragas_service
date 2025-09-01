from typing import Any, Dict, List, Optional

import cohere
import pandas as pd
from loguru import logger

from app.core.config import settings
from app.models.sample import SampleBase

# Import RAGAS components
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper

# Import LLM and embedding providers
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    answer_similarity,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.metrics._aspect_critic import coherence

# LangChain imports for different providers
try:
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import DashScopeEmbeddings
    OPENAI_AVAILABLE = True
    ANTHROPIC_AVAILABLE = True
    COHERE_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    ANTHROPIC_AVAILABLE = False
    COHERE_AVAILABLE = False


# debugging wrapper for LangchainEmbeddingsWrapper



class RAGASService:
    """Service for running RAGAS evaluations"""
    
    def __init__(self):
        self.metrics_map = {
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "faithfulness": faithfulness,
            "context_recall": context_recall,
            "answer_correctness": answer_correctness,
            "answer_similarity": answer_similarity,
            "critique_tone": coherence
        }
        self.llm = None
        self.embeddings = None
    
    def _setup_llm(self, llm_config: Optional[Dict[str, Any]]) -> None:
        """Setup LLM for evaluation"""
        if not llm_config:
            return None
        
        provider = llm_config.get("provider", settings.OPENAI_API_KEY)
        model = llm_config.get("model", settings.OPENAI_MODEL)
        api_key = llm_config.get("api_key", settings.OPENAI_API_KEY)
        base_url = llm_config.get("base_url", settings.OPENAI_BASE_URL)
        
        if not api_key:
            logger.warning("No API key provided for LLM")
            return None
        
        if provider == "openai" and OPENAI_AVAILABLE:
            self.llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, extra_body={"enable_thinking": False})
            self.llm = LangchainLLMWrapper(self.llm)
        elif provider == "anthropic" and ANTHROPIC_AVAILABLE:
            self.llm = ChatAnthropic(model_name=model, api_key=api_key, base_url=base_url, timeout=10, stop=["\n\n"])
            self.llm = LangchainLLMWrapper(self.llm)
        else:
            logger.warning(f"Unsupported LLM provider: {provider} or package not installed")
    
    def _setup_embeddings(self, embeddings_config: Optional[Dict[str, Any]]) -> Optional[Any]:
        """Setup embeddings for evaluation"""
        if embeddings_config:
            model = embeddings_config.get("model", settings.OPENAI_EMBEDDING_MODEL)
            api_key = embeddings_config.get("api_key", settings.OPENAI_API_KEY)
        else:
            model:str = settings.OPENAI_EMBEDDING_MODEL
            api_key = settings.OPENAI_API_KEY
        
        if not api_key:
            logger.warning("No API key provided for embeddings")
            return None
        
        # self.embeddings = OpenAIEmbeddings(model=model, api_key=api_key, base_url=base_url, dimensions=1024, tiktoken_enabled=False, tiktoken_model_name="text-embedding-ada-002")
        self.embeddings = DashScopeEmbeddings(
            dashscope_api_key=api_key,
            model=model, 
        )
        self.embeddings = LangchainEmbeddingsWrapper(self.embeddings)
    
    def _convert_sample_to_ragas_format(self, sample: dict[str, Any]) -> SampleBase:
        return SampleBase(**sample)
    
    async def evaluate_single_sample(
        self,
        sample: Dict[str, Any],
        metrics: List[Dict[str, Any]],
        llm_config: Optional[Dict[str, Any]] = None,
        embeddings_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate a single sample"""
        try:
            # Setup LLM and embeddings
            self._setup_llm(llm_config)
            self._setup_embeddings(embeddings_config)
            
            # Convert sample to RAGAS format
            ragas_sample = self._convert_sample_to_ragas_format(sample)
            
            # Create RAGAS dataset
            df = pd.DataFrame([ragas_sample.model_dump()])
            ragas_dataset = EvaluationDataset.from_pandas(df)
            
            # Select metrics
            selected_metrics = []
            for metric_config in metrics:
                metric_name = metric_config.get("name")
                if metric_name in self.metrics_map:
                    metric = self.metrics_map[metric_name]
                    selected_metrics.append(metric)
                else:
                    logger.warning(f"Unknown metric: {metric_name}")
            
            if not selected_metrics:
                raise ValueError("No valid metrics selected")
            
            # Run evaluation
            results = evaluate(
                ragas_dataset,
                metrics=selected_metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            # Extract scores from EvaluationResult
            scores = {}
            reasoning = {}
            cost = {"tokens": 0, "cost": 0.0, "currency": "USD"}
            
            # Access scores from the results object
            for metric_name in results._repr_dict.keys():
                if metric_name in self.metrics_map:
                    scores[metric_name] = float(results._repr_dict[metric_name])
            
            # Get individual sample scores if available
            if len(results.scores) > 0:
                sample_scores = results.scores[0]  # First sample
                for metric_name in sample_scores.keys():
                    if metric_name in self.metrics_map:
                        scores[metric_name] = float(sample_scores[metric_name])
            
            return {
                "scores": scores,
                "reasoning": reasoning if reasoning else None,
                "cost": cost
            }
            
        except Exception as e:
            logger.error(f"Error evaluating single sample: {e}")
            raise
    
    async def evaluate_batch(
        self,
        samples: List[Any],
        metrics: List[Dict[str, Any]],
        llm_config: Optional[Dict[str, Any]] = None,
        embeddings_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Evaluate a batch of samples"""
        try:
            # Setup LLM and embeddings
            
            
            # Convert samples to RAGAS format
            ragas_samples = []
            for sample in samples:
                ragas_sample = self._convert_sample_to_ragas_format(sample)
                ragas_samples.append(ragas_sample.model_dump())
            
            # Create RAGAS dataset
            df = pd.DataFrame(ragas_samples)
            ragas_dataset = EvaluationDataset.from_pandas(df)
            
            # Select metrics
            selected_metrics = []
            for metric_config in metrics:
                metric_name = metric_config.get("name")
                if metric_name in self.metrics_map:
                    metric = self.metrics_map[metric_name]
                    selected_metrics.append(metric)
                else:
                    logger.warning(f"Unknown metric: {metric_name}")
            
            if not selected_metrics:
                raise ValueError("No valid metrics selected")
            
            # Run evaluation
            results = evaluate(
                ragas_dataset,
                metrics=selected_metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            # Process results for each sample
            batch_results = []
            for i, sample in enumerate(samples):
                scores = {}
                reasoning = {}
                cost = {"tokens": 0, "cost": 0.0, "currency": "USD"}
                
                # Extract scores for this sample from results.scores
                if i < len(results.scores):
                    sample_scores = results.scores[i]
                    for metric_name in sample_scores.keys():
                        if metric_name in self.metrics_map:
                            scores[metric_name] = float(sample_scores[metric_name])
                else:
                    # Fallback to overall scores if individual scores not available
                    for metric_name in results._repr_dict.keys():
                        if metric_name in self.metrics_map:
                            scores[metric_name] = float(results._repr_dict[metric_name])
                
                batch_results.append({
                    "scores": scores,
                    "reasoning": reasoning if reasoning else None,
                    "cost": cost
                })
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Error evaluating batch: {e}")
            raise
    
    def get_available_metrics(self) -> List[Dict[str, Any]]:
        """Get list of available metrics with their configurations"""
        metrics_info = []
        
        for name, metric in self.metrics_map.items():
            metric_info = {
                "name": name,
                "description": self._get_metric_description(name),
                "type": self._get_metric_type(name),
                "supported_sample_types": ["single_turn", "multi_turn"],
                "parameters": self._get_metric_parameters(name),
                "default_config": self._get_metric_default_config(name),
                "example_usage": self._get_metric_example(name)
            }
            metrics_info.append(metric_info)
        
        return metrics_info
    
    def _get_metric_description(self, metric_name: str) -> str:
        """Get metric description"""
        descriptions = {
            "answer_relevancy": "Measures how relevant the answer is to the question",
            "context_precision": "Measures the precision of retrieved contexts",
            "faithfulness": "Measures how faithful the answer is to the provided context",
            "context_recall": "Measures the recall of retrieved contexts",
            "answer_correctness": "Measures the correctness of the answer",
            "answer_similarity": "Measures the similarity between generated and reference answers",
            "context_relevancy": "Measures the relevancy of retrieved contexts",
            "critique_tone": "Evaluates the tone and style of the answer"
        }
        return descriptions.get(metric_name, "No description available")
    
    def _get_metric_type(self, metric_name: str) -> str:
        """Get metric type"""
        llm_based = ["answer_relevancy", "faithfulness", "answer_correctness", "critique_tone"]
        embedding_based = ["context_precision", "context_recall", "answer_similarity", "context_relevancy"]
        
        if metric_name in llm_based:
            return "llm_based"
        elif metric_name in embedding_based:
            return "embedding_based"
        else:
            return "rule_based"
    
    def _get_metric_parameters(self, metric_name: str) -> Dict[str, Any]:
        """Get metric parameters"""
        return {
            "llm_required": metric_name in ["answer_relevancy", "faithfulness", "answer_correctness", "critique_tone"],
            "embeddings_required": metric_name in ["context_precision", "context_recall", "answer_similarity", "context_relevancy"]
        }
    
    def _get_metric_default_config(self, metric_name: str) -> Dict[str, Any]:
        """Get metric default configuration"""
        return {
            "llm": "gpt-4o" if metric_name in ["answer_relevancy", "faithfulness", "answer_correctness", "critique_tone"] else None,
            "embeddings": "text-embedding-3-small" if metric_name in ["context_precision", "context_recall", "answer_similarity", "context_relevancy"] else None
        }
    
    def _get_metric_example(self, metric_name: str) -> Dict[str, Any]:
        """Get metric example usage"""
        return {
            "sample": {
                "user_input": "What is the capital of France?",
                "response": "The capital of France is Paris.",
                "retrieved_contexts": ["Paris is the capital and largest city of France."]
            },
            "expected_score": 0.9
        }


# python -m app.services.ragas_service
if __name__ == "__main__":
    service = RAGASService()

    llm_config = {
        "provider": "openai",
        "model": "qwen3-4b",
        "api_key": settings.OPENAI_API_KEY,
        "base_url": settings.OPENAI_BASE_URL
    }

    all_metrics = service.get_available_metrics()
    print(all_metrics)

    sample = {
        "user_input": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "retrieved_contexts": [
            "Paris is the capital and largest city of France."
            "Beijing is the capital of China."
            ],
        "reference": "Paris is the capital of France."
    }

    async def main():

        service._setup_llm(llm_config)
        service._setup_embeddings(None)

        # test simple evaluation
        result = await service.evaluate_single_sample(sample, metrics=all_metrics)

        # test batch evaluation
        # result = await service.evaluate_batch(samples=[sample, sample], metrics=all_metrics)



        print(result)

    import asyncio
    asyncio.run(main())