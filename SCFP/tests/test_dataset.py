"""
Unit tests for SCFP dataset classes.
"""

import pytest
import json
import tempfile
import os
from transformers import DebertaV2Tokenizer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scfp.data.dataset import CorrectionTrace, FailureMode, SCFPDataset


class TestCorrectionTrace:
    """Test CorrectionTrace class."""
    
    def test_creation(self):
        """Test basic trace creation."""
        trace = CorrectionTrace(
            prompt="What is 2+2?",
            initial_response="5",
            critique="That's wrong, 2+2=4",
            final_response="4",
            failure_mode=FailureMode.JUSTIFICATION_HALLUCINATION,
            is_success=True
        )
        
        assert trace.prompt == "What is 2+2?"
        assert trace.initial_response == "5"
        assert trace.critique == "That's wrong, 2+2=4"
        assert trace.final_response == "4"
        assert trace.failure_mode == FailureMode.JUSTIFICATION_HALLUCINATION
        assert trace.is_success == True
    
    def test_to_input_text(self):
        """Test input text generation."""
        trace = CorrectionTrace(
            prompt="What is 2+2?",
            initial_response="5",
            critique="That's wrong, 2+2=4",
            final_response="4",
            failure_mode=FailureMode.SUCCESS,
            is_success=True
        )
        
        input_text = trace.to_input_text()
        assert "What is 2+2?" in input_text
        assert "5" in input_text
        assert "That's wrong, 2+2=4" in input_text
        assert "[SEP]" in input_text
    
    def test_to_dict_from_dict(self):
        """Test serialization and deserialization."""
        trace = CorrectionTrace(
            prompt="Test prompt",
            initial_response="Test response",
            critique="Test critique",
            final_response="Test final",
            failure_mode=FailureMode.CONFIDENCE_MISCALIBRATION,
            is_success=False,
            metadata={"domain": "test"}
        )
        
        # Convert to dict
        trace_dict = trace.to_dict()
        
        # Convert back to trace
        restored_trace = CorrectionTrace.from_dict(trace_dict)
        
        assert restored_trace.prompt == trace.prompt
        assert restored_trace.initial_response == trace.initial_response
        assert restored_trace.critique == trace.critique
        assert restored_trace.final_response == trace.final_response
        assert restored_trace.failure_mode == trace.failure_mode
        assert restored_trace.is_success == trace.is_success
        assert restored_trace.metadata == trace.metadata


class TestSCFPDataset:
    """Test SCFPDataset class."""
    
    @pytest.fixture
    def sample_traces(self):
        """Create sample traces for testing."""
        return [
            CorrectionTrace(
                prompt="What is the capital of France?",
                initial_response="London",
                critique="That's wrong, it's Paris",
                final_response="Paris",
                failure_mode=FailureMode.SUCCESS,
                is_success=True
            ),
            CorrectionTrace(
                prompt="Calculate 2+2",
                initial_response="5",
                critique="Let me recalculate: 2+2=4",
                final_response="4",
                failure_mode=FailureMode.JUSTIFICATION_HALLUCINATION,
                is_success=False
            )
        ]
    
    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer for testing."""
        return DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
    
    def test_dataset_creation(self, sample_traces, tokenizer):
        """Test dataset creation."""
        dataset = SCFPDataset(
            traces=sample_traces,
            tokenizer=tokenizer,
            max_length=512
        )
        
        assert len(dataset) == 2
        assert dataset.tokenizer == tokenizer
        assert dataset.max_length == 512
    
    def test_dataset_getitem(self, sample_traces, tokenizer):
        """Test dataset item retrieval."""
        dataset = SCFPDataset(
            traces=sample_traces,
            tokenizer=tokenizer,
            max_length=512
        )
        
        item = dataset[0]
        
        # Check required keys
        required_keys = ["input_ids", "attention_mask", "binary_label", "multiclass_label"]
        for key in required_keys:
            assert key in item
        
        # Check data types and shapes
        assert len(item["input_ids"]) <= 512
        assert len(item["attention_mask"]) <= 512
        assert len(item["input_ids"]) == len(item["attention_mask"])
        assert isinstance(item["binary_label"], int)
        assert isinstance(item["multiclass_label"], int)
    
    def test_dataset_from_json(self, sample_traces, tokenizer):
        """Test loading dataset from JSON."""
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([trace.to_dict() for trace in sample_traces], f)
            temp_path = f.name
        
        try:
            # Load dataset from JSON
            dataset = SCFPDataset.from_json(
                temp_path,
                tokenizer=tokenizer,
                max_length=512
            )
            
            assert len(dataset) == 2
            
            # Test first item
            item = dataset[0]
            assert "input_ids" in item
            assert "binary_label" in item
            
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_failure_mode_mapping(self, sample_traces, tokenizer):
        """Test failure mode to index mapping."""
        dataset = SCFPDataset(
            traces=sample_traces,
            tokenizer=tokenizer,
            max_length=512
        )
        
        # First trace is success (should map to 0)
        item0 = dataset[0]
        assert item0["multiclass_label"] == 0
        assert item0["binary_label"] == 1  # Success
        
        # Second trace is failure (should map to failure mode index)
        item1 = dataset[1]
        assert item1["multiclass_label"] == 1  # JH maps to 1
        assert item1["binary_label"] == 0  # Failure


class TestFailureMode:
    """Test FailureMode enum."""
    
    def test_failure_modes(self):
        """Test all failure modes are defined."""
        expected_modes = [
            "success", "jh", "cm", "ba", "oc", "rm"
        ]
        
        for mode in expected_modes:
            assert hasattr(FailureMode, mode.upper())
    
    def test_mode_values(self):
        """Test failure mode values."""
        assert FailureMode.SUCCESS.value == "success"
        assert FailureMode.JUSTIFICATION_HALLUCINATION.value == "jh"
        assert FailureMode.CONFIDENCE_MISCALIBRATION.value == "cm"
        assert FailureMode.BIAS_AMPLIFICATION.value == "ba"
        assert FailureMode.OVER_CORRECTION.value == "oc"
        assert FailureMode.REASONING_MYOPIA.value == "rm"


if __name__ == "__main__":
    pytest.main([__file__])
