from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
    ensembler,
)
from ragas.run_config import RunConfig
from ragas.dataset_schema import SingleTurnSample
from ragas.prompt import PydanticPrompt

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


class QCA(BaseModel):
    """Question, Context, Answer input model"""
    question: str
    context: str


class ContextRecallClassification(BaseModel):
    """Individual context recall classification"""
    statement: str
    reason: str
    attributed: int


class ContextRecallClassifications(BaseModel):
    """List of context recall classifications"""
    classifications: t.List[ContextRecallClassification]


def weight(ith:int, N:int):
    return (N - ith) / N

def weight_exponential(ith:int, N:int, lambda_:float=0.11):
    return np.exp(-lambda_ * ith / N)  # Adding a 0.5 factor to make decay more gentle

def mrr_score(responses: t.List[ContextRecallClassification], total_contexts: int) -> float:
    """
    Calculate MRR (Mean Reciprocal Rank) score.
    Returns 1 - (first_relevant_position / total_contexts)
    If first relevant position is 0, score = 1 - 0 = 1
    If first relevant position is 3 out of 5, score = 1 - 3/5 = 0.4
    If no relevant documents, score = 0
    If no contexts provided (total_contexts = 0), score = 0
    """
    if total_contexts == 0:
        return 0.0
    
    # Find the first relevant document (attributed=1)
    for i, response in enumerate(responses):
        if response.attributed:
            # MRR score: 1 - (position / total_contexts)
            return 1.0 - (i / total_contexts)
    
    # No relevant documents found
    return 0.0



class ContextRecallClassificationPrompt(
    PydanticPrompt[QCA, ContextRecallClassifications]
):
    """Prompt for evaluating context recall using the same pattern as RAGAS"""
    name: str = "context_recall_classification"
    instruction: str = """
    Given a question and multiple retrieved contexts, analyze each context to determine if it is relevant and useful for answering the question.
    
    For each context, classify it as:
    - Use 'Yes' (1) if the context is relevant and provides useful information for the question
    - Use 'No' (0) if the context is completely irrelevant or doesn't help answer the question
    
    Consider:
    1. RELEVANCE: Is the context related to the question topic? (Even indirectly related contexts can be useful)
    2. COMPLETENESS: Does the context provide useful information? (Doesn't need to be comprehensive)
    3. INFORMATION DENSITY: Does the context contain relevant details?
    4. COVERAGE: Does the context address aspects related to the question?
    
    Be more lenient in your evaluation - if the context provides any useful information related to the question topic, mark it as relevant (1).
    Only mark as irrelevant (0) if the context is completely unrelated to the question.
    
    Output JSON with classifications for each context, including the reason for each classification.
    """
    input_model = QCA
    output_model = ContextRecallClassifications
    examples = [
        (
            QCA(
                question="What is the capital of France?",
                context="Context 1: Paris is the capital and largest city of France.\nContext 2: Paris is located in the north-central part of France.\nContext 3: The Eiffel Tower is a famous landmark in Paris."
            ),
            ContextRecallClassifications(
                classifications=[
                    ContextRecallClassification(
                        statement="Context 1: Paris is the capital and largest city of France.",
                        reason="This context directly answers the question about France's capital.",
                        attributed=1,
                    ),
                    ContextRecallClassification(
                        statement="Context 2: Paris is located in the north-central part of France.",
                        reason="This context provides relevant geographical information about Paris, which is the capital of France.",
                        attributed=1,
                    ),
                    ContextRecallClassification(
                        statement="Context 3: The Eiffel Tower is a famous landmark in Paris.",
                        reason="This context provides relevant information about Paris, which is the capital of France.",
                        attributed=1,
                    ),
                ]
            ),
        ),
        (
            QCA(
                question="What is the weather like today?",
                context="Context 1: The capital of France is Paris.\nContext 2: Python is a programming language.\nContext 3: The sky is blue but this is not weather information."
            ),
            ContextRecallClassifications(
                classifications=[
                    ContextRecallClassification(
                        statement="Context 1: The capital of France is Paris.",
                        reason="This context is completely irrelevant to the question about weather.",
                        attributed=0,
                    ),
                    ContextRecallClassification(
                        statement="Context 2: Python is a programming language.",
                        reason="This context is completely irrelevant to the question about weather.",
                        attributed=0,
                    ),
                    ContextRecallClassification(
                        statement="Context 3: The sky is blue but this is not weather information.",
                        reason="This context mentions the sky but explicitly states it's not weather information, making it irrelevant.",
                        attributed=0,
                    ),
                ]
            ),
        ),
    ]


@dataclass
class CustomContextRecallMetric(MetricWithLLM, SingleTurnMetric):
    """
    Custom metric to evaluate context recall using only user_input and retrieved_contexts.
    
    This metric measures how well the retrieved contexts cover the information needed
    to answer the user query, focusing on relevance and completeness.
    Uses LLM for intelligent evaluation following the RAGAS pattern.
    """
    
    name: str = "custom_context_recall"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "retrieved_contexts"}
        }
    )
    output_type = MetricOutputType.CONTINUOUS
    context_recall_prompt: PydanticPrompt = field(
        default_factory=ContextRecallClassificationPrompt
    )
    max_retries: int = 1
    
    def _compute_score(self, responses: t.List[ContextRecallClassification], total_contexts: int) -> float:
        """Compute the final score using MRR (Mean Reciprocal Rank)"""
        if total_contexts == 0:
            return 0.0
        
        if len(responses) != total_contexts:
            logger.warning(f"Mismatch between responses ({len(responses)}) and total contexts ({total_contexts})")
            # Fallback to simple average if there's a mismatch
            relevant_count = sum(1 for item in responses if item.attributed)
            return relevant_count / total_contexts if total_contexts > 0 else 0.0
        
        # Use MRR scoring
        score = mrr_score(responses, total_contexts)

        if np.isnan(score):
            logger.warning("The LLM did not return a valid classification.")

        return score
    
    def init(self, run_config: RunConfig):
        """Initialize the metric with run configuration"""
        super().init(run_config)
        
        # Validate that LLM is available for this metric
        if not self.llm:
            logger.warning("LLM not initialized for custom_context_recall metric. LLM-based evaluation will not be available.")
        else:
            logger.info("LLM initialized for custom_context_recall metric")
    
    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        """Calculate the context recall score for a single sample (RAGAS format)"""
        row = sample.to_dict()
        return await self._ascore(row, callbacks)
    
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        """Calculate the context recall score using LLM classification"""
        assert self.llm is not None, "LLM must be set before use"
        
        try:
            # Extract data from the row
            user_input = row.get("user_input", "")
            retrieved_contexts = row.get("retrieved_contexts", [])
            
            # Handle case where user_input might be a list of messages
            if isinstance(user_input, list):
                # Extract text content from messages
                user_input = " ".join([msg.get('content', '') if isinstance(msg, dict) else str(msg) for msg in user_input])
            
            if not user_input or not retrieved_contexts:
                logger.debug(f"Empty user_input or retrieved_contexts: user_input='{user_input}', retrieved_contexts={retrieved_contexts}")
                return 0.0
            
            # Ensure retrieved_contexts is a list
            if retrieved_contexts is None:
                retrieved_contexts = []
            
            total_contexts = len(retrieved_contexts)
            logger.debug(f"Calculating recall for user_input: '{user_input[:100]}...' and {total_contexts} contexts")
            
            # Format contexts for single LLM call
            formatted_contexts = []
            for i, context in enumerate(retrieved_contexts, 1):
                formatted_contexts.append(f"Context {i}: {context}")
            
            combined_context = "\n".join(formatted_contexts)
            logger.debug(f"Evaluating {total_contexts} contexts in single call")
            
            # Run classification for all contexts at once
            classifications_list: t.List[
                ContextRecallClassifications
            ] = await self.context_recall_prompt.generate_multiple(
                data=QCA(
                    question=user_input,
                    context=combined_context,
                ),
                llm=self.llm,
                callbacks=callbacks,
            )
            
            # Process classifications following RAGAS pattern
            classification_dicts = []
            for classification in classifications_list:
                classification_dicts.append(
                    [clasif.model_dump() for clasif in classification.classifications]
                )

            # Ensemble the classifications
            ensembled_clasif = ensembler.from_discrete(classification_dicts, "attributed")
            
            # Convert to ContextRecallClassification objects
            all_classifications = [
                ContextRecallClassification(**clasif) for clasif in ensembled_clasif
            ]
            
            # Compute the final score using total contexts as denominator
            score = self._compute_score(all_classifications, total_contexts)
            
            logger.debug(f"Calculated recall score: {score} ({sum(1 for c in all_classifications if c.attributed)}/{total_contexts} relevant)")
            return float(score) if not np.isnan(score) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating context recall: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0


# Create an instance of the custom metric
custom_context_recall = CustomContextRecallMetric()


# python -m app.services.custom_matric
if __name__ == "__main__":
    import asyncio
    from app.core.config import settings
    
    async def test_metric():
        """Test the custom context recall metric"""
        print("Testing Custom Context Recall Metric...")
        
        # Initialize the metric
        metric = CustomContextRecallMetric()
        
        # Set up LLM (you'll need to configure this based on your setup)
        try:
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper
            
            # Create LangChain LLM
            langchain_llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                temperature=0,
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL
            )
            
            # Wrap with RAGAS LLM wrapper
            metric.llm = LangchainLLMWrapper(langchain_llm)
            print("✓ LLM initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize LLM: {e}")
            print("Please set OPENAI_API_KEY environment variable")
            return
        
        # Test cases
        test_cases = [
            {
                "name": "Test 1: All relevant contexts (first position)",
                "user_input": "What is the capital of France?",
                "retrieved_contexts": [
                    "Paris is the capital and largest city of France.",  # Position 0, relevant
                    "Paris is located in the north-central part of France.",
                    "The Eiffel Tower is a famous landmark in Paris."
                ],
                "expected_score": 1.0  # MRR: 1 - 0/3 = 1.0
            },
            {
                "name": "Test 2: Relevant context at position 1",
                "user_input": "What is the capital of France?",
                "retrieved_contexts": [
                    "Beijing is the capital of China.",                   # Position 0, irrelevant
                    "Paris is the capital and largest city of France.",  # Position 1, relevant
                    "Python is a programming language.",                  # Position 2, irrelevant
                    "Paris is located in the north-central part of France."  # Position 3, relevant
                ],
                "expected_score": 0.75  # MRR: 1 - 1/4 = 0.75
            },
            {
                "name": "Test 3: Relevant context at position 2",
                "user_input": "What is the capital of France?",
                "retrieved_contexts": [
                    "Beijing is the capital of China.",                   # Position 0, irrelevant
                    "Python is a programming language.",                  # Position 1, irrelevant
                    "Paris is the capital and largest city of France.",  # Position 2, relevant
                    "Paris is located in the north-central part of France."  # Position 3, relevant
                ],
                "expected_score": 0.5  # MRR: 1 - 2/4 = 0.5
            },
            {
                "name": "Test 4: All irrelevant contexts",
                "user_input": "What is the weather like today?",
                "retrieved_contexts": [
                    "The capital of France is Paris.",
                    "Python is a programming language.",
                    "The sky is blue but this is not weather information."
                ],
                "expected_score": 0.0  # No relevant contexts found
            },
            {
                "name": "Test 5: Empty contexts",
                "user_input": "What is the capital of France?",
                "retrieved_contexts": [],
                "expected_score": 0.0  # No contexts to evaluate - returns 0.0 instead of NaN
            }
        ]
        
        # Run tests
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- {test_case['name']} ---")
            print(f"Question: {test_case['user_input']}")
            print(f"Contexts ({len(test_case['retrieved_contexts'])}):")
            for j, ctx in enumerate(test_case['retrieved_contexts'], 1):
                print(f"  {j}. {ctx}")
            
            try:
                # Calculate score
                score = await metric._ascore(test_case, callbacks=None)
                expected = test_case['expected_score']
                
                print(f"Calculated Score: {score:.2f}")
                print(f"Expected Score: {expected:.2f}")
                
                # Check if score is close to expected (allow some tolerance for LLM variability)
                if abs(score - expected) < 0.2:
                    print("✓ Test PASSED")
                else:
                    print("✗ Test FAILED - Score differs significantly from expected")
                    
            except Exception as e:
                print(f"✗ Test FAILED with error: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*50)
        print("Test Summary:")
        print("The metric evaluates retrieved contexts using MRR (Mean Reciprocal Rank).")
        print("Score = 1 - (first_relevant_position / total_contexts)")
        print("Examples:")
        print("- First relevant at position 0: score = 1 - 0/N = 1.0")
        print("- First relevant at position 2 out of 5: score = 1 - 2/5 = 0.6")
        print("- No relevant contexts: score = 0.0")
        print("Score range: 0.0 (no relevant contexts) to 1.0 (first context is relevant)")
        print("Benefits:")
        print("- Single LLM call for all contexts (reduces token usage)")
        print("- MRR rewards early placement of relevant documents")
        print("- More efficient and cost-effective evaluation")
    
    # Run the test
    asyncio.run(test_metric())
    
