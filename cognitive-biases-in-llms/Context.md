# Context.md - Cognitive Biases in LLMs Project

## Project Overview

This project implements a comprehensive framework for evaluating cognitive biases in Large Language Models (LLMs). It provides systematic testing, data generation, and analysis capabilities for 30 different cognitive biases across 20+ state-of-the-art LLMs.

## Higher-Level Project Structure

```
cognitive-biases-in-llms/
├── core/                           # Core framework components
├── models/                         # LLM interfaces for different providers
├── tests/                          # Individual cognitive bias test definitions
├── data/                           # Generated datasets and results
├── run/                           # Experiment execution scripts
├── plots/                         # Analysis visualization outputs
├── demo.py                        # Simple demonstration script
├── run_analysis.py                # Main analysis execution (moved from run/)
├── run_randomness_analysis.py     # Randomness analysis (moved from run/)
├── similarity_analysis_copy_32_2050.py  # Core analysis functions (moved from run/)
└── requirements.txt               # Python dependencies
```

## Core Module (`core/`)

### `base.py`
**Main Classes:**
- `LLM`: Abstract base class for all language models
  - `prompt()`: Send prompts to LLM and get responses
  - `populate()`: Fill template gaps using LLM generation
  - `decide()`: Make decisions on test cases
  - `decide_all()`: Batch decision processing
- `TestGenerator`: Abstract base for bias-specific test generators
  - `generate()`: Create test cases for specific bias
  - `generate_all()`: Batch test case generation
  - `sample_custom_values()`: Sample bias-specific parameters
- `RatioScaleMetric`: Quantitative bias measurement for ratio scale data
  - `compute()`: Calculate bias scores for test results
  - `aggregate()`: Aggregate individual scores into overall bias measure
- `NominalScaleMetric`: Quantitative bias measurement for nominal scale data
- `AggregationMetric`: Combine multiple bias measurements

**Exception Classes:**
- `PopulationError`: Raised during test case template population
- `DecisionError`: Raised during decision-making process
- `MetricCalculationError`: Raised during bias metric calculation

### `utils.py`
**Main Functions:**
- `get_generator(bias: str)`: Dynamically load test generator for specific bias
- `get_metric(bias: str)`: Dynamically load metric calculator for specific bias
- `get_model(model_name: str)`: Instantiate LLM interface for specified model
- `get_supported_models()`: Return list of all supported model names

**Constants:**
- `SUPPORTED_MODELS`: List of 45+ supported LLM models from 8 providers

### `testing.py`
**Main Classes:**
- `TestCase`: Container for control/treatment test pair
- `Template`: Template with gaps for LLM population
- `TestConfig`: Configuration loader for bias-specific settings
- `DecisionResult`: Container for LLM decision outputs

### `add_test.py`
**Main Functions:**
- Interactive script for adding new cognitive bias tests
- Creates directory structure and template files

## Models Module (`models/`)

### Structure by Provider
- `OpenAI/`: GPT models (GPT-4o, GPT-4o-Mini, GPT-3.5-Turbo)
- `Meta/`: Llama models (3.1 and 3.2 series)
- `Google/`: Gemini models (1.5 Flash, Pro)
- `Anthropic/`: Claude models (3.5 Sonnet, Haiku)
- `MistralAI/`: Mistral models (Large-2, Small)
- `Microsoft/`: WizardLM models
- `Alibaba/`: Qwen models
- `ZeroOneAI/`: Yi models
- `Random/`: Random baseline model
- `Local/`: Local model interfaces

Each provider directory contains:
- `model.py`: Model class implementations inheriting from `LLM`
- `prompts.yml`: Standardized prompts for generation and decision tasks

## Tests Module (`tests/`)

### Structure
Each cognitive bias has its own directory with:
- `test.py`: Bias-specific `TestGenerator` and `Metric` classes
- `config.xml`: Test templates and custom value definitions
- `__init__.py`: Module initialization

### Implemented Biases (30 total)
- **Anchoring**: Initial value influences subsequent judgments
- **AvailabilityHeuristic**: Probability judgments based on memory accessibility
- **ConfirmationBias**: Seeking information confirming existing beliefs
- **FramingEffect**: Different presentations of same information affect decisions
- **LossAversion**: Stronger preference for avoiding losses than acquiring gains
- **EndowmentEffect**: Overvaluing owned items
- **StatusQuoBias**: Preference for current state of affairs
- **OptimismBias**: Overestimating positive outcomes
- **PlanningFallacy**: Underestimating time, costs, and risks
- **HindsightBias**: "I knew it all along" effect
- **Plus 20 additional biases**

## Analysis Files (Recently Moved to Root)

### `similarity_analysis_copy_32_2050.py`
**Main Functions:**
- `load_decision_data()`: Load and process LLM decision results
- `load_model_bias_data()`: Calculate bias scores for each model
- `run_hierarchical_clustering_analysis()`: Perform clustering analysis
- `plot_clustering_analysis()`: Generate visualization plots
- `perform_unsupervised_clustering_analysis()`: K-means and other clustering
- `perform_supervised_clustering_analysis()`: Supervised learning analysis
- `analyze_label_agreement()`: Compare clustering results
- `impute_missing_values()`: Handle missing data

### `run_analysis.py`
**Main Functions:**
- `run_analysis_load_and_prepare_data()`: Data loading and preprocessing
- `run_hierarchical_clustering()`: Execute hierarchical clustering
- `run_both_clustering_analyses()`: Combined supervised/unsupervised analysis
- `save_clustering_results()`: Persist analysis results
- `main()`: Orchestrate full analysis pipeline

### `run_randomness_analysis.py`
**Main Functions:**
- Analysis of randomness patterns in model responses
- Statistical testing of bias consistency

## Data Module (`data/`)

### Structure
- `generated_datasets/`: CSV files with generated test cases per bias
- `generated_tests/`: XML files with individual test instances
- `decision_results/`: Model decision outputs organized by model
- `generation_logs/`: Detailed logs from test generation process
- `checked_datasets/`: Manually validated test cases
- `scenarios.txt`: 200 decision-making scenarios
- `full_dataset.csv`: Combined dataset with all biases

## Run Module (`run/`)

### Key Scripts
- `scenario_generation.py`: Generate decision-making scenarios
- `test_generation.py`: Generate test case instances
- `test_decision.py`: Obtain LLM decisions on test cases
- `dataset_assembly.py`: Combine individual datasets
- `test_check.py`: Manual validation of generated tests
- `analysis.py`: Analysis pipeline execution
- `analysis.ipynb`: Interactive analysis notebook

## General Workflows

### 1. Test Case Generation Workflow
```
1. Load scenarios from scenarios.txt
2. For each bias:
   a. Load bias-specific TestGenerator
   b. Sample custom values (anchors, options, etc.)
   c. Generate control/treatment template pairs
   d. Populate templates using generation LLM
   e. Validate generated test cases
   f. Save to generated_tests/ and generated_datasets/
```

### 2. Decision Collection Workflow
```
1. Load full_dataset.csv with all test cases
2. For each target LLM:
   a. Batch test cases for parallel processing
   b. Present each test case to model
   c. Extract decision from model response
   d. Handle errors and retries
   e. Save results to decision_results/{model}/
```

### 3. Analysis Workflow
```
1. Load decision results for all models
2. Calculate bias scores using bias-specific metrics
3. Filter and clean data (handle missing values)
4. Perform clustering analysis:
   a. Hierarchical clustering by bias similarity
   b. Supervised clustering by known model groups
   c. Unsupervised clustering to discover patterns
5. Generate visualizations and statistical reports
6. Save results and plots
```

### 4. Adding New Bias Workflow
```
1. Run core/add_test.py
2. Edit tests/{BiasName}/config.xml with templates
3. Implement tests/{BiasName}/test.py with:
   a. {BiasName}TestGenerator class
   b. {BiasName}Metric class
4. Add bias to test generation scripts
5. Generate and validate test cases
```

## Key Dependencies

- **OpenAI API**: GPT model access
- **Anthropic API**: Claude model access  
- **Google Generative AI**: Gemini model access
- **HuggingFace Transformers**: Local model inference
- **scikit-learn**: Clustering and machine learning
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn**: Visualization
- **PyYAML**: Configuration management
- **tqdm**: Progress tracking

## Research Contributions

1. **Systematic Framework**: General-purpose testing framework for cognitive biases
2. **Comprehensive Dataset**: 30,000 test cases across 30 biases and 200 scenarios
3. **Multi-Model Evaluation**: 20 state-of-the-art LLMs from 8 providers
4. **Reproducible Pipeline**: Complete workflow for bias evaluation research 