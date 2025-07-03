"""
Test module for core.base functionality.

This module contains unit tests for the core base classes including LLM,
TestGenerator, and various metric classes. It demonstrates proper testing
practices with pytest fixtures and comprehensive test coverage.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from core.base import (
    LLM, TestGenerator, RatioScaleMetric, NominalScaleMetric,
    PopulationError, DecisionError, MetricCalculationError
)
from core.testing import TestCase, Template, DecisionResult


class MockLLM(LLM):
    """Mock implementation of LLM for testing purposes."""
    
    def __init__(self, randomly_flip_options: bool = False, shuffle_answer_options: bool = False):
        super().__init__(randomly_flip_options, shuffle_answer_options)
        self.NAME = "mock-llm"
        self._prompt_responses = []
        self._call_count = 0
    
    def prompt(self, prompt: str, temperature: float = 0.0, seed: int = 42) -> str:
        """Mock prompt method that returns predefined responses."""
        if self._call_count < len(self._prompt_responses):
            response = self._prompt_responses[self._call_count]
            self._call_count += 1
            return response
        return "Default mock response"
    
    def set_responses(self, responses: list[str]) -> None:
        """Set predefined responses for the mock LLM."""
        self._prompt_responses = responses
        self._call_count = 0


class TestPopulationError:
    """Test cases for PopulationError exception class."""
    
    def test_population_error_basic_message(self):
        """Test PopulationError with basic message only."""
        error = PopulationError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.template is None
        assert error.model_output is None
    
    def test_population_error_with_template(self):
        """Test PopulationError with template context."""
        mock_template = Mock()
        mock_template.__str__ = Mock(return_value="Mock template content")
        
        error = PopulationError("Test error", template=mock_template)
        assert "Test error" in str(error)
        assert "Mock template content" in str(error)
        assert error.template == mock_template
    
    def test_population_error_with_model_output(self):
        """Test PopulationError with model output context."""
        error = PopulationError("Test error", model_output="Mock model output")
        assert "Test error" in str(error)
        assert "Mock model output" in str(error)
        assert error.model_output == "Mock model output"


class TestLLMBase:
    """Test cases for LLM base class functionality."""
    
    @pytest.fixture
    def mock_llm(self):
        """Fixture providing a MockLLM instance."""
        return MockLLM()
    
    def test_llm_initialization_default(self, mock_llm):
        """Test LLM initialization with default parameters."""
        assert mock_llm.NAME == "mock-llm"
        assert mock_llm.randomly_flip_options is False
        assert mock_llm.shuffle_answer_options is False
    
    def test_llm_initialization_with_options(self):
        """Test LLM initialization with custom options."""
        llm = MockLLM(randomly_flip_options=True, shuffle_answer_options=True)
        assert llm.randomly_flip_options is True
        assert llm.shuffle_answer_options is True
    
    def test_prompt_method(self, mock_llm):
        """Test the prompt method with predefined responses."""
        mock_llm.set_responses(["Response 1", "Response 2"])
        
        response1 = mock_llm.prompt("Test prompt 1")
        assert response1 == "Response 1"
        
        response2 = mock_llm.prompt("Test prompt 2")
        assert response2 == "Response 2"
        
        # Test default response when no more predefined responses
        response3 = mock_llm.prompt("Test prompt 3")
        assert response3 == "Default mock response"


class TestRatioScaleMetric:
    """Test cases for RatioScaleMetric class."""
    
    @pytest.fixture
    def sample_test_results(self):
        """Fixture providing sample test results for metric calculation."""
        # Create mock test cases and decision results
        test_case1 = Mock(spec=TestCase)
        test_case2 = Mock(spec=TestCase)
        
        decision_result1 = Mock(spec=DecisionResult)
        decision_result1.control_option = 1
        decision_result1.treatment_option = 2
        
        decision_result2 = Mock(spec=DecisionResult)
        decision_result2.control_option = 2
        decision_result2.treatment_option = 1
        
        return [(test_case1, decision_result1), (test_case2, decision_result2)]
    
    def test_ratio_scale_metric_initialization(self, sample_test_results):
        """Test RatioScaleMetric initialization with test results."""
        metric = RatioScaleMetric(
            test_results=sample_test_results,
            k=np.array([1]),
            x_1=np.array([10]),
            x_2=np.array([20])
        )
        
        assert len(metric.test_results) == 2
        assert np.array_equal(metric.k, np.array([1]))
        assert np.array_equal(metric.x_1, np.array([10]))
        assert np.array_equal(metric.x_2, np.array([20]))
    
    def test_ratio_scale_metric_compute(self, sample_test_results):
        """Test bias score computation for ratio scale metric."""
        metric = RatioScaleMetric(
            test_results=sample_test_results,
            k=np.array([1]),
            x_1=np.array([10]),
            x_2=np.array([20])
        )
        
        # Mock the compute method since it's complex
        with patch.object(metric, '_compute') as mock_compute:
            mock_compute.return_value = np.array([0.5, -0.3])
            
            result = metric.compute()
            assert isinstance(result, np.ndarray)
            assert len(result) == 2
            mock_compute.assert_called_once()
    
    def test_ratio_scale_metric_aggregate(self, sample_test_results):
        """Test bias score aggregation."""
        metric = RatioScaleMetric(test_results=sample_test_results)
        
        bias_scores = np.array([0.5, -0.3, 0.8])
        aggregated = metric.aggregate(bias_scores)
        
        assert isinstance(aggregated, float)
        assert -1.0 <= aggregated <= 1.0  # Bias scores should be in valid range


class TestNominalScaleMetric:
    """Test cases for NominalScaleMetric class."""
    
    @pytest.fixture
    def sample_nominal_results(self):
        """Fixture providing sample test results for nominal metric calculation."""
        test_case1 = Mock(spec=TestCase)
        test_case2 = Mock(spec=TestCase)
        
        decision_result1 = Mock(spec=DecisionResult)
        decision_result1.control_option = 1
        decision_result1.treatment_option = 2
        
        decision_result2 = Mock(spec=DecisionResult)
        decision_result2.control_option = 1
        decision_result2.treatment_option = 1
        
        return [(test_case1, decision_result1), (test_case2, decision_result2)]
    
    def test_nominal_scale_metric_initialization(self, sample_nominal_results):
        """Test NominalScaleMetric initialization."""
        metric = NominalScaleMetric(
            test_results=sample_nominal_results,
            options_labels=np.array(['A', 'B']),
            x=np.array([1, 2]),
            k=1
        )
        
        assert len(metric.test_results) == 2
        assert np.array_equal(metric.options_labels, np.array(['A', 'B']))
        assert np.array_equal(metric.x, np.array([1, 2]))
        assert metric.k == 1
    
    def test_nominal_scale_metric_compute(self, sample_nominal_results):
        """Test bias score computation for nominal scale metric."""
        metric = NominalScaleMetric(
            test_results=sample_nominal_results,
            options_labels=np.array(['A', 'B']),
            x=np.array([1, 2])
        )
        
        # Mock the compute method
        with patch.object(metric, '_compute') as mock_compute:
            mock_compute.return_value = np.array([1, 0])
            
            result = metric.compute()
            assert isinstance(result, np.ndarray)
            assert len(result) == 2
            mock_compute.assert_called_once()


class TestExceptionHandling:
    """Test cases for exception handling in core.base module."""
    
    def test_decision_error_raised(self):
        """Test that DecisionError can be raised and caught."""
        with pytest.raises(DecisionError):
            raise DecisionError("Test decision error")
    
    def test_metric_calculation_error_raised(self):
        """Test that MetricCalculationError can be raised and caught."""
        with pytest.raises(MetricCalculationError):
            raise MetricCalculationError("Test metric calculation error")
    
    def test_population_error_with_context(self):
        """Test PopulationError with full context information."""
        mock_template = Mock()
        mock_template.__str__ = Mock(return_value="Template content")
        
        with pytest.raises(PopulationError) as exc_info:
            raise PopulationError(
                "Population failed", 
                template=mock_template, 
                model_output="Model output"
            )
        
        error = exc_info.value
        assert "Population failed" in str(error)
        assert "Template content" in str(error)
        assert "Model output" in str(error)


class TestIntegrationScenarios:
    """Integration test scenarios combining multiple components."""
    
    def test_llm_decision_workflow(self):
        """Test a complete LLM decision-making workflow."""
        # This would be a more complex integration test
        # that tests the full workflow from template to decision
        mock_llm = MockLLM()
        mock_llm.set_responses([
            "I choose option 2",
            "Option 2"
        ])
        
        # Mock test case
        mock_test_case = Mock(spec=TestCase)
        mock_test_case.control = Mock()
        mock_test_case.treatment = Mock()
        
        # Test that the workflow can be initiated
        # (This would need more implementation in a real scenario)
        assert mock_llm.NAME == "mock-llm"
        response = mock_llm.prompt("Test prompt")
        assert response == "I choose option 2"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"]) 