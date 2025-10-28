"""
LLM-based automated labeling for active learning.

Uses Langchain to:
1. Label instances (documents/texts)
2. Label features (words/terms) for each class
3. Provide confidence scores
4. Explain reasoning

This replaces human annotation in the active learning loop.
"""

from typing import List, Dict, Tuple, Optional
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import logging
import json

logger = logging.getLogger(__name__)


# Pydantic models for structured output
class InstanceLabelOutput(BaseModel):
    """Structured output for instance labeling."""
    label: str = Field(description="The predicted label/class for this instance")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation of why this label was chosen")


class FeatureLabelOutput(BaseModel):
    """Structured output for feature labeling."""
    relevant: bool = Field(description="Whether this feature is relevant to the class")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation")


class LLMLabeler:
    """
    LLM-based labeler for active learning.

    This class uses an LLM to automatically label instances and features,
    replacing human annotation in the active learning loop.
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        api_key: Optional[str] = None
    ):
        """
        Initialize the LLM labeler.

        Args:
            model_name: Name of the LLM model to use
            temperature: Sampling temperature (lower = more deterministic)
            api_key: Optional API key (uses env var if not provided)
        """
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=api_key
        )
        self.temperature = temperature

        logger.info(f"Initialized LLM labeler with model: {model_name}")

    def label_instance(
        self,
        text: str,
        labels: List[str],
        task_description: str,
        few_shot_examples: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, float, str]:
        """
        Label a single text instance.

        Args:
            text: The text to label
            labels: List of possible labels
            task_description: Description of the classification task
            few_shot_examples: Optional list of example {"text": "...", "label": "..."}

        Returns:
            Tuple of (predicted_label, confidence, reasoning)
        """
        # Build the prompt
        system_prompt = self._build_instance_labeling_system_prompt(
            labels,
            task_description
        )

        human_prompt = self._build_instance_labeling_human_prompt(
            text,
            few_shot_examples
        )

        try:
            # Call LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]

            response = self.llm(messages)
            content = response.content

            # Parse response
            try:
                # Try to parse as JSON
                result = json.loads(content)
                label = result.get("label", labels[0])
                confidence = result.get("confidence", 0.5)
                reasoning = result.get("reasoning", "")
            except json.JSONDecodeError:
                # Fallback: extract label from text
                label = self._extract_label_from_text(content, labels)
                confidence = 0.7
                reasoning = content

            logger.info(f"Labeled instance as '{label}' (confidence: {confidence:.2f})")

            return label, confidence, reasoning

        except Exception as e:
            logger.error(f"Instance labeling failed: {e}")
            # Return first label as fallback
            return labels[0], 0.5, f"Error: {str(e)}"

    def label_feature(
        self,
        feature: str,
        label: str,
        task_description: str,
        context_examples: Optional[List[str]] = None
    ) -> Tuple[bool, float, str]:
        """
        Determine if a feature is relevant for a given label.

        Args:
            feature: The feature/term to evaluate
            label: The label to check relevance for
            task_description: Description of the classification task
            context_examples: Optional list of example texts for this label

        Returns:
            Tuple of (is_relevant, confidence, reasoning)
        """
        system_prompt = self._build_feature_labeling_system_prompt(
            label,
            task_description
        )

        human_prompt = self._build_feature_labeling_human_prompt(
            feature,
            context_examples
        )

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]

            response = self.llm(messages)
            content = response.content

            # Parse response
            try:
                result = json.loads(content)
                relevant = result.get("relevant", False)
                confidence = result.get("confidence", 0.5)
                reasoning = result.get("reasoning", "")
            except json.JSONDecodeError:
                # Fallback: check for yes/no/relevant keywords
                content_lower = content.lower()
                if any(word in content_lower for word in ["yes", "relevant", "indicative"]):
                    relevant = True
                    confidence = 0.7
                else:
                    relevant = False
                    confidence = 0.7
                reasoning = content

            logger.info(f"Feature '{feature}' for label '{label}': "
                       f"relevant={relevant} (confidence: {confidence:.2f})")

            return relevant, confidence, reasoning

        except Exception as e:
            logger.error(f"Feature labeling failed: {e}")
            return False, 0.5, f"Error: {str(e)}"

    def label_batch_instances(
        self,
        texts: List[str],
        labels: List[str],
        task_description: str,
        few_shot_examples: Optional[List[Dict[str, str]]] = None
    ) -> List[Tuple[str, float, str]]:
        """
        Label a batch of instances.

        Args:
            texts: List of texts to label
            labels: List of possible labels
            task_description: Description of the task
            few_shot_examples: Optional few-shot examples

        Returns:
            List of (label, confidence, reasoning) tuples
        """
        results = []

        for text in texts:
            result = self.label_instance(
                text,
                labels,
                task_description,
                few_shot_examples
            )
            results.append(result)

        return results

    def _build_instance_labeling_system_prompt(
        self,
        labels: List[str],
        task_description: str
    ) -> str:
        """Build system prompt for instance labeling."""
        labels_str = ", ".join([f"'{l}'" for l in labels])

        prompt = f"""You are an expert text classifier helping with an active learning task.

TASK: {task_description}

LABELS: {labels_str}

Your job is to:
1. Carefully read and analyze each text
2. Classify it into one of the given labels
3. Provide a confidence score (0.0 to 1.0)
4. Briefly explain your reasoning

Respond ONLY with a JSON object in this exact format:
{{
    "label": "<one of the given labels>",
    "confidence": <number between 0 and 1>,
    "reasoning": "<brief explanation>"
}}

Be concise but accurate. Consider:
- Keywords and terminology specific to each category
- Context and semantic meaning
- Overall topic and intent

Do not include any text outside the JSON object."""

        return prompt

    def _build_instance_labeling_human_prompt(
        self,
        text: str,
        few_shot_examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Build human prompt for instance labeling."""
        prompt_parts = []

        # Add few-shot examples if provided
        if few_shot_examples:
            prompt_parts.append("Here are some examples:\n")
            for i, example in enumerate(few_shot_examples, 1):
                prompt_parts.append(f"Example {i}:")
                prompt_parts.append(f"Text: {example['text'][:200]}")
                prompt_parts.append(f"Label: {example['label']}\n")

        # Add the text to classify
        prompt_parts.append("Now classify this text:\n")
        prompt_parts.append(f"Text: {text}\n")
        prompt_parts.append("Respond with the JSON object:")

        return "\n".join(prompt_parts)

    def _build_feature_labeling_system_prompt(
        self,
        label: str,
        task_description: str
    ) -> str:
        """Build system prompt for feature labeling."""
        prompt = f"""You are an expert in feature selection for text classification.

TASK: {task_description}

LABEL: '{label}'

Your job is to determine whether a given word/feature is RELEVANT and INDICATIVE of the label '{label}'.

A feature is relevant if:
- It frequently appears in documents of this class
- It is discriminative (helps distinguish this class from others)
- It carries semantic meaning related to this class

Respond ONLY with a JSON object in this exact format:
{{
    "relevant": <true or false>,
    "confidence": <number between 0 and 1>,
    "reasoning": "<brief explanation>"
}}

Be strict and conservative. Only mark features as relevant if they are truly indicative."""

        return prompt

    def _build_feature_labeling_human_prompt(
        self,
        feature: str,
        context_examples: Optional[List[str]] = None
    ) -> str:
        """Build human prompt for feature labeling."""
        prompt_parts = []

        # Add context examples if provided
        if context_examples and len(context_examples) > 0:
            prompt_parts.append("Here are some example texts from this class:\n")
            for i, example in enumerate(context_examples[:3], 1):
                prompt_parts.append(f"{i}. {example[:150]}...")

            prompt_parts.append("")

        # Add the feature to evaluate
        prompt_parts.append(f"Is the word/feature '{feature}' relevant and indicative of this class?")
        prompt_parts.append("\nRespond with the JSON object:")

        return "\n".join(prompt_parts)

    @staticmethod
    def _extract_label_from_text(text: str, labels: List[str]) -> str:
        """Extract label from LLM response text (fallback)."""
        text_lower = text.lower()

        # Try to find label in response
        for label in labels:
            if label.lower() in text_lower:
                return label

        # Default to first label
        return labels[0]

    def generate_initial_labeled_features(
        self,
        labels: List[str],
        task_description: str,
        n_features_per_label: int = 10
    ) -> Dict[str, List[str]]:
        """
        Generate initial feature labels for cold start.

        Uses LLM to suggest discriminative features for each label.

        Args:
            labels: List of class labels
            task_description: Description of the task
            n_features_per_label: Number of features to generate per label

        Returns:
            Dictionary mapping label to list of relevant features
        """
        feature_dict = {}

        for label in labels:
            prompt = f"""You are helping with text classification.

TASK: {task_description}

For the class/label '{label}', suggest {n_features_per_label} words or terms that are:
1. Highly indicative of this class
2. Discriminative (help distinguish from other classes)
3. Likely to appear in texts of this class

Respond with ONLY a JSON list of words, like:
["word1", "word2", "word3", ...]

Keep words lowercase and simple (no phrases)."""

            try:
                response = self.llm([HumanMessage(content=prompt)])
                content = response.content

                # Parse JSON
                features = json.loads(content)

                if isinstance(features, list):
                    feature_dict[label] = features[:n_features_per_label]
                else:
                    feature_dict[label] = []

            except Exception as e:
                logger.error(f"Failed to generate features for '{label}': {e}")
                feature_dict[label] = []

        logger.info(f"Generated initial features for {len(labels)} labels")

        return feature_dict
