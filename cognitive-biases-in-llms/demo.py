"""
Demonstration script for the cognitive biases in LLMs framework.

This script demonstrates the basic workflow of generating a test case for a
cognitive bias and obtaining a decision result from an LLM. It showcases
the end-to-end process from scenario selection to bias metric calculation.
"""

import logging
import random
from typing import List

from core.utils import get_generator, get_metric
from models.OpenAI.model import GptThreePointFiveTurbo, GptFourO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define a cognitive bias to test (provide the name in Pascal Case, e.g., 'IllusionOfControl')
BIAS: str = 'Anchoring'               

# Define other execution parameters
TEMPERATURE_GENERATION: float = 0.7     # LLM temperature applied when generating test cases
TEMPERATURE_DECISION: float = 0.0       # LLM temperature applied when deciding test cases
RANDOMLY_FLIP_OPTIONS: bool = True      # Whether answer option order will be randomly flipped in 50% of test cases
SHUFFLE_ANSWER_OPTIONS: bool = False    # Whether answer options will be randomly shuffled for all test cases


def load_scenarios(scenarios_file: str = 'data/scenarios.txt') -> List[str]:
    """
    Load decision-making scenarios from file.

    Args:
        scenarios_file (str): Path to the scenarios file.

    Returns:
        List[str]: List of scenario strings.

    Raises:
        FileNotFoundError: If the scenarios file doesn't exist.
    """
    try:
        with open(scenarios_file, 'r') as f:
            scenarios = f.readlines()
        logger.info(f"Loaded {len(scenarios)} scenarios from {scenarios_file}")
        return scenarios
    except FileNotFoundError:
        logger.error(f"Scenarios file not found: {scenarios_file}")
        raise


def run_bias_demonstration(
    bias: str = BIAS,
    temperature_generation: float = TEMPERATURE_GENERATION,
    temperature_decision: float = TEMPERATURE_DECISION,
    randomly_flip_options: bool = RANDOMLY_FLIP_OPTIONS,
    shuffle_answer_options: bool = SHUFFLE_ANSWER_OPTIONS
) -> None:
    """
    Run a complete demonstration of bias testing workflow.

    Args:
        bias (str): Name of the cognitive bias to test.
        temperature_generation (float): Temperature for test case generation.
        temperature_decision (float): Temperature for decision making.
        randomly_flip_options (bool): Whether to randomly flip answer options.
        shuffle_answer_options (bool): Whether to shuffle answer options.

    Returns:
        None

    Raises:
        ImportError: If the specified bias test generator or metric is not found.
        Exception: If any step in the workflow fails.
    """
    logger.info(f"Starting bias demonstration for: {bias}")
    
    try:
        # Load the pre-defined scenario strings
        scenarios = load_scenarios()

        # Randomly pick a scenario
        scenario = random.choice(scenarios).strip()
        logger.info(f"Selected scenario: {scenario[:100]}...")

        # Sample a random seed
        seed = random.randint(0, 1000)
        logger.info(f"Using random seed: {seed}")
        
        # Load the test generator and metric for the bias
        logger.info(f"Loading test generator and metric for {bias}")
        generator = get_generator(bias)
        metric_class = get_metric(bias)

        # Instantiate the generation and decision LLMs
        logger.info("Initializing LLM models")
        generation_model = GptFourO()
        decision_model = GptThreePointFiveTurbo(randomly_flip_options, shuffle_answer_options)

        # Generate one test case instance for the scenario
        logger.info("Generating test case")
        test_cases = generator.generate_all(
            generation_model, 
            [scenario], 
            temperature_generation, 
            seed, 
            num_instances=1, 
            max_retries=5
        )
        logger.info(f"Generated {len(test_cases)} test case(s)")
        print(f"\nGenerated test case:\n{test_cases}")

        # Obtain a decision result for the generated test case
        logger.info("Obtaining decision results")
        decision_results = decision_model.decide_all(test_cases, temperature_decision, seed)
        logger.info(f"Obtained {len(decision_results)} decision result(s)")
        print(f"\nDecision results:\n{decision_results}")

        # Calculate the bias score
        logger.info("Calculating bias metrics")
        metric = metric_class(test_results=list(zip(test_cases, decision_results)))
        computed_metric = metric.compute()
        aggregated_metric = metric.aggregate(computed_metric)
        
        print(f'\nBias metric per case:\n{computed_metric}')
        print(f'Aggregated bias metric: {aggregated_metric}')
        
        logger.info(f"Demonstration completed successfully. Final bias score: {aggregated_metric}")
        
    except ImportError as e:
        logger.error(f"Failed to load bias test components: {e}")
        raise
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    """
    Main execution point for the demonstration script.
    
    This demonstrates the complete workflow for testing cognitive biases in LLMs:
    1. Load scenarios from file
    2. Generate test cases using generation LLM
    3. Obtain decisions using decision LLM
    4. Calculate bias metrics
    """
    try:
        run_bias_demonstration()
    except Exception as e:
        logger.error(f"Demo execution failed: {e}")
        exit(1)