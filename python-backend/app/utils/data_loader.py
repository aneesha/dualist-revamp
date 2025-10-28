"""
Data loading utilities for handling various input formats.

Supports:
- ZIP archives with folder-based labels
- Text files
- JSON formats
- CSV formats

Based on the original Java Util.java implementation.
"""

import os
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import csv
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and process training/test data."""

    @staticmethod
    def load_from_zip(
        zip_path: str,
        pipe,
        extract_to: Optional[str] = None
    ) -> Tuple[List[Dict[str, int]], List[str], List[str]]:
        """
        Load data from ZIP archive with folder-based labels.

        Expected structure:
        archive.zip/
          label1/
            doc1.txt
            doc2.txt
          label2/
            doc3.txt

        Args:
            zip_path: Path to ZIP file
            pipe: Text processing pipeline
            extract_to: Optional directory to extract to (temp if None)

        Returns:
            Tuple of (X, y, instance_ids)
            - X: List of feature dictionaries
            - y: List of labels
            - instance_ids: List of instance identifiers
        """
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")

        X = []
        y = []
        instance_ids = []

        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get all file paths
            file_list = zf.namelist()

            for file_path in file_list:
                # Skip directories and hidden files
                if file_path.endswith('/') or '/..' in file_path:
                    continue

                # Skip files in root directory
                parts = Path(file_path).parts
                if len(parts) < 2:
                    continue

                # Label is the top-level folder name
                label = parts[0]

                # Read file content
                try:
                    content = zf.read(file_path).decode('utf-8', errors='ignore')
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
                    continue

                # Process through pipeline
                features = pipe.process(content)

                if len(features) > 0:
                    X.append(features)
                    y.append(label)
                    instance_ids.append(file_path)

        logger.info(f"Loaded {len(X)} instances from {zip_path}")
        logger.info(f"Labels: {set(y)}")

        return X, y, instance_ids

    @staticmethod
    def load_from_directory(
        dir_path: str,
        pipe,
        recursive: bool = True
    ) -> Tuple[List[Dict[str, int]], List[str], List[str]]:
        """
        Load data from directory with folder-based labels.

        Expected structure:
        directory/
          label1/
            doc1.txt
            doc2.txt
          label2/
            doc3.txt

        Args:
            dir_path: Path to directory
            pipe: Text processing pipeline
            recursive: Whether to search recursively

        Returns:
            Tuple of (X, y, instance_ids)
        """
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        X = []
        y = []
        instance_ids = []

        base_path = Path(dir_path)

        # Iterate through label directories
        for label_dir in base_path.iterdir():
            if not label_dir.is_dir():
                continue

            label = label_dir.name

            # Find all text files
            if recursive:
                files = list(label_dir.rglob('*.txt'))
            else:
                files = list(label_dir.glob('*.txt'))

            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    features = pipe.process(content)

                    if len(features) > 0:
                        X.append(features)
                        y.append(label)
                        instance_ids.append(str(file_path.relative_to(base_path)))

                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")

        logger.info(f"Loaded {len(X)} instances from {dir_path}")
        logger.info(f"Labels: {set(y)}")

        return X, y, instance_ids

    @staticmethod
    def load_from_text_file(
        file_path: str,
        pipe,
        label: str,
        line_by_line: bool = False
    ) -> Tuple[List[Dict[str, int]], List[str], List[str]]:
        """
        Load data from a single text file.

        Args:
            file_path: Path to text file
            pipe: Text processing pipeline
            label: Label to assign to all instances
            line_by_line: If True, treat each line as separate instance

        Returns:
            Tuple of (X, y, instance_ids)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        X = []
        y = []
        instance_ids = []

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            if line_by_line:
                for i, line in enumerate(f):
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    features = pipe.process(line)
                    if len(features) > 0:
                        X.append(features)
                        y.append(label)
                        instance_ids.append(f"{file_path}:line{i+1}")
            else:
                content = f.read()
                features = pipe.process(content)
                if len(features) > 0:
                    X.append(features)
                    y.append(label)
                    instance_ids.append(file_path)

        logger.info(f"Loaded {len(X)} instances from {file_path}")

        return X, y, instance_ids

    @staticmethod
    def load_from_json(
        json_path: str,
        pipe,
        text_field: str = "text",
        label_field: str = "label",
        id_field: Optional[str] = "id"
    ) -> Tuple[List[Dict[str, int]], List[str], List[str]]:
        """
        Load data from JSON file.

        Expected format:
        [
          {"text": "...", "label": "...", "id": "..."},
          ...
        ]

        Args:
            json_path: Path to JSON file
            pipe: Text processing pipeline
            text_field: Name of text field
            label_field: Name of label field
            id_field: Name of ID field (optional)

        Returns:
            Tuple of (X, y, instance_ids)
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File not found: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        X = []
        y = []
        instance_ids = []

        for i, item in enumerate(data):
            if text_field not in item or label_field not in item:
                logger.warning(f"Skipping item {i}: missing required fields")
                continue

            text = item[text_field]
            label = item[label_field]
            item_id = item.get(id_field, f"item_{i}") if id_field else f"item_{i}"

            features = pipe.process(text)
            if len(features) > 0:
                X.append(features)
                y.append(label)
                instance_ids.append(item_id)

        logger.info(f"Loaded {len(X)} instances from {json_path}")
        logger.info(f"Labels: {set(y)}")

        return X, y, instance_ids

    @staticmethod
    def load_from_csv(
        csv_path: str,
        pipe,
        text_column: str = "text",
        label_column: str = "label",
        id_column: Optional[str] = None
    ) -> Tuple[List[Dict[str, int]], List[str], List[str]]:
        """
        Load data from CSV file.

        Args:
            csv_path: Path to CSV file
            pipe: Text processing pipeline
            text_column: Name of text column
            label_column: Name of label column
            id_column: Name of ID column (optional)

        Returns:
            Tuple of (X, y, instance_ids)
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")

        X = []
        y = []
        instance_ids = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                if text_column not in row or label_column not in row:
                    logger.warning(f"Skipping row {i}: missing required columns")
                    continue

                text = row[text_column]
                label = row[label_column]
                item_id = row.get(id_column, f"row_{i}") if id_column else f"row_{i}"

                features = pipe.process(text)
                if len(features) > 0:
                    X.append(features)
                    y.append(label)
                    instance_ids.append(item_id)

        logger.info(f"Loaded {len(X)} instances from {csv_path}")
        logger.info(f"Labels: {set(y)}")

        return X, y, instance_ids

    @staticmethod
    def save_predictions_to_csv(
        predictions: List[Tuple[str, str, float]],
        output_path: str
    ):
        """
        Save predictions to CSV file.

        Args:
            predictions: List of (instance_id, predicted_label, confidence)
            output_path: Path to output CSV
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['instance_id', 'predicted_label', 'confidence'])
            writer.writerows(predictions)

        logger.info(f"Saved {len(predictions)} predictions to {output_path}")
