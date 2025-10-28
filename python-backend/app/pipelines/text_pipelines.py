"""
Text processing pipelines for different data types.

Implements:
- DocumentPipe: Process full documents
- SimpleLinesPipe: Process line-by-line with bigrams
- TwitterPipe: Process tweets with emoticons, mentions, hashtags
- EntityPipe: Process named entities with orthographic features

Based on the original Java pipe implementations.
"""

import re
from typing import List, Dict, Set, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class BasePipe:
    """Base class for text processing pipelines."""

    def __init__(self, stopwords: Optional[Set[str]] = None):
        """
        Initialize the pipeline.

        Args:
            stopwords: Set of stopwords to remove (optional)
        """
        self.stopwords = stopwords or self._default_stopwords()

    @staticmethod
    def _default_stopwords() -> Set[str]:
        """Default English stopwords."""
        return {
            'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
            'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
            'between', 'both', 'but', 'by', 'could', 'did', 'do', 'does', 'doing', 'down',
            'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have',
            'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his',
            'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me',
            'might', 'more', 'most', 'must', 'my', 'myself', 'no', 'nor', 'not', 'now',
            'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves',
            'out', 'over', 'own', 's', 'same', 'she', 'should', 'so', 'some', 'such',
            't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
            'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under',
            'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
            'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'you', 'your',
            'yours', 'yourself', 'yourselves'
        }

    def process(self, text: str) -> Dict[str, int]:
        """
        Process text and return feature counts.

        Args:
            text: Input text

        Returns:
            Dictionary of feature -> count
        """
        raise NotImplementedError


class DocumentPipe(BasePipe):
    """
    Process whole documents as single instances.

    Features:
    - Unigrams (single words)
    - Lowercase normalization
    - HTML removal
    - Number replacement
    - Stopword removal
    """

    def __init__(self, stopwords: Optional[Set[str]] = None):
        super().__init__(stopwords)

    @staticmethod
    def _remove_html(text: str) -> str:
        """Remove HTML tags and entities."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        text = re.sub(r'&#\d+;', ' ', text)
        return text

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Replace numbers with "00"
        text = re.sub(r'\d+', '00', text)
        # Extract words (Unicode letters and marks)
        tokens = re.findall(r'[\w]+', text, re.UNICODE)
        return [t.lower() for t in tokens if len(t) > 0]

    def process(self, text: str) -> Dict[str, int]:
        """
        Process document text into feature counts.

        Args:
            text: Document text

        Returns:
            Dictionary of feature -> count
        """
        # Remove HTML
        text = self._remove_html(text)

        # Tokenize
        tokens = self._tokenize(text)

        # Remove stopwords and count
        tokens = [t for t in tokens if t not in self.stopwords]

        return dict(Counter(tokens))


class SimpleLinesPipe(BasePipe):
    """
    Process text line-by-line with unigrams and bigrams.

    Features:
    - Unigrams
    - Bigrams
    - Lowercase normalization
    """

    def __init__(self, use_bigrams: bool = True):
        super().__init__(stopwords=set())  # No stopword removal
        self.use_bigrams = use_bigrams

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Replace numbers
        text = re.sub(r'\d+', '00', text)
        # Extract words
        tokens = re.findall(r'[\w]+', text, re.UNICODE)
        return [t.lower() for t in tokens if len(t) > 0]

    def _extract_bigrams(self, tokens: List[str]) -> List[str]:
        """Extract bigrams from tokens."""
        if len(tokens) < 2:
            return []
        return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]

    def process(self, text: str) -> Dict[str, int]:
        """
        Process text into unigrams and bigrams.

        Args:
            text: Input text

        Returns:
            Dictionary of feature -> count
        """
        tokens = self._tokenize(text)

        features = []
        features.extend(tokens)

        if self.use_bigrams:
            features.extend(self._extract_bigrams(tokens))

        return dict(Counter(features))


class TwitterPipe(BasePipe):
    """
    Process tweets with social media features.

    Features:
    - Standard tokens
    - Emoticons (preserved)
    - @mentions
    - #hashtags
    - Negation handling
    - Bigrams
    """

    def __init__(self):
        super().__init__(stopwords=set())

        # Common emoticons
        self.emoticons = [
            ':-)', ':)', ':D', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
            ':^)', ':-D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D', '=-3', '=3',
            ':-(', ':(', ':c', ':<', ':[', '>:[', ':{', '>:(', ':-[', ':-<',
            '<3', '</3'
        ]

        # Negation words
        self.negations = {
            'no', 'not', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere',
            "n't", 'dont', "don't", 'doesnt', "doesn't", 'didnt', "didn't",
            'wont', "won't", 'wouldnt', "wouldn't", 'cant', "can't", 'couldnt', "couldn't"
        }

    def _extract_features(self, text: str) -> List[str]:
        """Extract Twitter-specific features."""
        features = []

        # Preserve emoticons
        for emoticon in self.emoticons:
            if emoticon in text:
                features.append(f"EMOTICON_{emoticon}")
                text = text.replace(emoticon, ' ')

        # Extract @mentions
        mentions = re.findall(r'@\w+', text)
        features.extend([f"MENTION_{m[1:]}" for m in mentions])

        # Extract #hashtags
        hashtags = re.findall(r'#\w+', text)
        features.extend([f"HASHTAG_{h[1:]}" for h in hashtags])

        # Remove mentions and hashtags from text
        text = re.sub(r'@\w+', ' ', text)
        text = re.sub(r'#\w+', ' ', text)

        # Tokenize remaining text
        text = re.sub(r'\d+', '00', text)
        tokens = re.findall(r"[\w']+", text, re.UNICODE)
        tokens = [t.lower() for t in tokens if len(t) > 0]

        # Handle negation
        in_negation = False
        processed_tokens = []

        for token in tokens:
            if token in self.negations:
                in_negation = True
                processed_tokens.append(token)
            elif in_negation:
                processed_tokens.append(f"NEG_{token}")
                # End negation at punctuation
                if token in {'.', '!', '?', ',', ';'}:
                    in_negation = False
            else:
                processed_tokens.append(token)

        features.extend(processed_tokens)

        # Add bigrams
        if len(processed_tokens) >= 2:
            bigrams = [
                f"{processed_tokens[i]}_{processed_tokens[i+1]}"
                for i in range(len(processed_tokens) - 1)
            ]
            features.extend(bigrams)

        return features

    def process(self, text: str) -> Dict[str, int]:
        """
        Process tweet text into features.

        Args:
            text: Tweet text

        Returns:
            Dictionary of feature -> count
        """
        features = self._extract_features(text)
        return dict(Counter(features))


class EntityPipe(BasePipe):
    """
    Process named entities with orthographic features.

    Features:
    - Entity text (lowercased)
    - Shape features (capitalization patterns)
    - Character type features
    - Prefix/suffix features
    """

    def __init__(self):
        super().__init__(stopwords=set())

    @staticmethod
    def _get_shape(word: str) -> str:
        """Get capitalization shape of word."""
        if word.isupper():
            return "ALL_CAPS"
        elif word.islower():
            return "all_lower"
        elif word[0].isupper() and word[1:].islower():
            return "Title_Case"
        elif any(c.isupper() for c in word) and any(c.islower() for c in word):
            return "MiXeD_CaSe"
        else:
            return "other"

    @staticmethod
    def _has_digit(word: str) -> bool:
        """Check if word contains digits."""
        return any(c.isdigit() for c in word)

    @staticmethod
    def _has_punctuation(word: str) -> bool:
        """Check if word contains punctuation."""
        return any(not c.isalnum() for c in word)

    def _get_affixes(self, word: str) -> List[str]:
        """Get prefix and suffix features."""
        features = []
        word_lower = word.lower()

        # Prefixes (2-4 chars)
        for n in range(2, min(5, len(word))):
            features.append(f"PREFIX_{word_lower[:n]}")

        # Suffixes (2-4 chars)
        for n in range(2, min(5, len(word))):
            features.append(f"SUFFIX_{word_lower[-n:]}")

        return features

    def process(self, text: str) -> Dict[str, int]:
        """
        Process entity text into orthographic features.

        Args:
            text: Entity text (may be tab-delimited with context)

        Returns:
            Dictionary of feature -> count
        """
        features = []

        # Split by tabs (entity + context)
        parts = text.split('\t')
        entity = parts[0].strip()

        # Split entity into words
        words = entity.split()

        for word in words:
            if len(word) == 0:
                continue

            # Add word itself
            features.append(word.lower())

            # Shape features
            features.append(f"SHAPE_{self._get_shape(word)}")

            # Has digit
            if self._has_digit(word):
                features.append("HAS_DIGIT")

            # Has punctuation
            if self._has_punctuation(word):
                features.append("HAS_PUNCT")

            # Affixes
            features.extend(self._get_affixes(word))

            # Length category
            if len(word) <= 3:
                features.append("LEN_SHORT")
            elif len(word) <= 7:
                features.append("LEN_MEDIUM")
            else:
                features.append("LEN_LONG")

        return dict(Counter(features))


def get_pipe(pipe_type: str) -> BasePipe:
    """
    Factory function to get a pipeline by type.

    Args:
        pipe_type: One of "document", "lines", "twitter", "entity"

    Returns:
        Pipeline instance
    """
    pipes = {
        "document": DocumentPipe,
        "lines": SimpleLinesPipe,
        "twitter": TwitterPipe,
        "entity": EntityPipe
    }

    if pipe_type not in pipes:
        raise ValueError(f"Unknown pipe type: {pipe_type}. Choose from {list(pipes.keys())}")

    return pipes[pipe_type]()
