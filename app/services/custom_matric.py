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


class ContextRecallClassificationPrompt(
    PydanticPrompt[QCA, ContextRecallClassifications]
):
    """Prompt for evaluating context recall using the same pattern as RAGAS"""
    name: str = "context_recall_classification"
    instruction: str = """
    Given a question and retrieved contexts, analyze how well the contexts cover the information needed to answer the question.
    
    For each context, classify if it can be attributed to answering the question:
    - Use 'Yes' (1) if the context is relevant and provides useful information for the question
    - Use 'No' (0) if the context is irrelevant or doesn't help answer the question
    
    Consider:
    1. RELEVANCE: Is the context directly related to the question topic?
    2. COMPLETENESS: Does the context provide comprehensive information?
    3. INFORMATION DENSITY: Is the context rich in useful details?
    4. COVERAGE: Does the context address the specific aspects asked in the question?
    
    Output JSON with reason for each classification.
    """
    input_model = QCA
    output_model = ContextRecallClassifications
    examples = [
        (
            QCA(
                question="What is the capital of France?",
                context="Paris is the capital and largest city of France. Paris is located in the north-central part of France. The Eiffel Tower is a famous landmark in Paris."
            ),
            ContextRecallClassifications(
                classifications=[
                    ContextRecallClassification(
                        statement="Paris is the capital and largest city of France.",
                        reason="Directly answers the question about France's capital.",
                        attributed=1,
                    ),
                    ContextRecallClassification(
                        statement="Paris is located in the north-central part of France.",
                        reason="Provides relevant geographical context about the capital.",
                        attributed=1,
                    ),
                    ContextRecallClassification(
                        statement="The Eiffel Tower is a famous landmark in Paris.",
                        reason="Adds cultural context and confirms Paris as the location.",
                        attributed=1,
                    ),
                ]
            ),
        ),
        (
            QCA(
                question="What is the weather like today?",
                context="The capital of France is Paris. Python is a programming language. The sky is blue."
            ),
            ContextRecallClassifications(
                classifications=[
                    ContextRecallClassification(
                        statement="The capital of France is Paris.",
                        reason="Completely irrelevant to weather information.",
                        attributed=0,
                    ),
                    ContextRecallClassification(
                        statement="Python is a programming language.",
                        reason="Completely irrelevant to weather information.",
                        attributed=0,
                    ),
                    ContextRecallClassification(
                        statement="The sky is blue.",
                        reason="Partially relevant to weather but insufficient information.",
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
    
    def _compute_score(self, responses: t.List[ContextRecallClassification]) -> float:
        """Compute the final score from classifications"""
        response = [1 if item.attributed else 0 for item in responses]
        denom = len(response)
        numerator = sum(response)
        score = numerator / denom if denom > 0 else np.nan

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
            
            logger.debug(f"Calculating recall for user_input: '{user_input[:100]}...' and {len(retrieved_contexts)} contexts")
            
            # Run classification using the RAGAS pattern
            classifications_list: t.List[
                ContextRecallClassifications
            ] = await self.context_recall_prompt.generate_multiple(
                data=QCA(
                    question=user_input,
                    context="\n".join(retrieved_contexts),
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
            
            # Compute the final score
            score = self._compute_score(
                [ContextRecallClassification(**clasif) for clasif in ensembled_clasif]
            )
            
            logger.debug(f"Calculated recall score: {score}")
            return float(score) if not np.isnan(score) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating context recall: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0.0


# Create an instance of the custom metric
custom_context_recall = CustomContextRecallMetric()
