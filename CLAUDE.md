# Multimodal TimesFM

## Overview

This project aims to extend Google's [TimesFM](https://github.com/google-research/timesfm) to support multimodal inputs including text by creating a wrapper class and fine-tuning it on the [Time-MMD](https://github.com/AdityaLab/Time-MMD) dataset.

## Project Structure

```
multimodal-timesfm/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── multimodal_timesfm.py    # Main wrapper class
│   │   └── text_encoder.py          # Text encoding components
│   ├── data/
│   │   ├── __init__.py
│   │   ├── time_mmd_dataset.py      # Time-MMD dataset loader
│   │   └── preprocessing.py         # Data preprocessing utilities
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Training logic
│   │   └── fine_tuner.py            # Fine-tuning specific code
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluator.py             # Evaluation metrics
│       └── ablation_study.py        # Ablation study implementation
├── scripts/
│   ├── setup_data.py                # Download and prepare Time-MMD data
│   ├── validate_wrapper.py          # Wrapper class validation script
│   ├── train.py                     # Main training script
│   ├── evaluate.py                  # Evaluation script
│   └── run_ablation.py              # Ablation study runner
├── configs/
│   ├── base_config.yaml             # Base configuration
│   ├── training_config.yaml         # Training parameters
│   └── model_config.yaml            # Model architecture settings
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_data.py
│   └── test_training.py
├── .python-version                  # Python 3.11
├── pyproject.toml                   # uv configuration
├── README.md
└── CLAUDE.md                        # This file
```

## Development Roadmap

### Phase 1: Project Setup and Basic Infrastructure

1. **Initialize project structure**
   - Set up uv project with Python 3.11.13
   - Create basic directory structure
   - Install TimesFM and dependencies
2. **Create wrapper class skeleton**
   - Implement `MultimodalTimesFM` wrapper class
   - Initially delegate all functionality to underlying TimesFM model
   - Define interfaces for multimodal input handling
3. **Data pipeline setup**
   - Implement Time-MMD dataset loader
   - Create data preprocessing utilities
   - Set up train/test split (70/30)
4. **Validation of basic functionality**
   - Test wrapper class with Time-MMD time series data
   - Verify forward pass and gradient computation
   - Validate training loop integration
   - Ensure consistent behavior with original TimesFM
   - Run baseline performance tests

### Phase 2: Model Architecture Enhancement

1. **Text encoding components**
   - Implement text encoder (e.g., using sentence transformers)
   - Design fusion mechanism for time series and text features
   - Integrate text features into TimesFM architecture
2. **Wrapper class enhancement**
   - Extend wrapper to handle text inputs
   - Implement forward pass with multimodal inputs
   - Add training-specific methods

### Phase 3: Training and Fine-tuning

1. **Training pipeline**
   - Implement training loop for multimodal inputs
   - Set up checkpointing and logging
   - Configure hyperparameters
2. **Fine-tuning implementation**
   - Fine-tune on 70% of Time-MMD dataset
   - Implement curriculum learning if beneficial
   - Monitor training metrics and convergence

### Phase 4: Evaluation and Ablation Study

1. **Evaluation framework**
   - Implement evaluation metrics for time series forecasting
   - Create comparison utilities
2. **Ablation study**
   - Train baseline TimesFM on time series data only
   - Train multimodal version with text information
   - Compare performance across various metrics
   - Generate comprehensive comparison reports

## Key Implementation Details

### Dependencies

- **timesfm**: Google's Time Series Foundation Model
- **torch**: PyTorch for neural network operations

### Model Architecture Considerations

- **Text Encoder**: Use pre-trained sentence transformers (e.g., all-MiniLM-L6-v2)
- **Feature Fusion**: Concatenation or attention-based fusion of text and time series features
- **Architecture Integration**: Minimal modification to TimesFM's core architecture
- **Training Strategy**: Freeze TimesFM weights initially, then fine-tune end-to-end

### Configuration Management

Use YAML files to manage:

- Model hyperparameters
- Training configuration
- Data preprocessing parameters
- Evaluation settings

### Testing Strategy

- Unit tests for model components
- Integration tests for data pipeline
- End-to-end tests for training pipeline
- Performance benchmarks

## Expected Deliverables

1. **Functional multimodal TimesFM wrapper class**
2. **Complete training and evaluation pipeline**
3. **Ablation study comparing:**
   - TimesFM with time series only
   - Multimodal TimesFM with text and time series
4. **Performance metrics and analysis**
5. **Documentation and reproducibility guidelines**

## Success Criteria

- [x] Wrapper class successfully extends TimesFM functionality
- [x] Wrapper class validation passes with Time-MMD dataset
- [x] Forward pass produces expected output shapes and ranges
- [x] Training loop integrates smoothly with wrapper class
- [x] Model can process both time series and text inputs
- [x] Text encoder and fusion mechanisms implemented and tested
- [x] Comprehensive preprocessing utilities for multimodal data
- [x] Full test suite covering all multimodal components
- [ ] Training pipeline completes without errors
- [ ] Ablation study shows meaningful performance comparison
- [ ] Results demonstrate the value (or lack thereof) of text information
- [ ] Code is well-documented and reproducible

## Next Steps

**Phase 2: Model Architecture Enhancement**

1. - [ ] Implement text encoder using sentence transformers
2. - [ ] Design fusion mechanism for time series and text features  
3. - [ ] Extend MultimodalTimesFM to handle text inputs
4. - [ ] Add multimodal forward pass functionality
5. - [ ] Create text preprocessing utilities
6. - [ ] Add tests for multimodal components

**Phase 3: Training Pipeline**

7. - [ ] Implement training loop for multimodal inputs
8. - [ ] Set up checkpointing and logging
9. - [ ] Configure hyperparameters for fine-tuning

**Phase 4: Evaluation**

10. - [ ] Implement evaluation metrics
11. - [ ] Create ablation study framework
12. - [ ] Generate performance comparison reports

## Notes

- Consider using mixed precision training for efficiency
- Implement proper error handling for multimodal input validation
- Document any architectural changes made to accommodate multimodal inputs
- Keep detailed logs of hyperparameter choices and their impact

## Bash Commands

- `uv run mypy .`: Type checking
- `uv run ruff check`: Linting
- `uv run ruff format`: Code formatting
- `uv run pytest tests/ -v`: Run test suite

## Documentation Conventions

- Use docstrings for all functions, classes, and modules
- Include type hints for all function parameters and return values
- Follow Google-style docstring format
- Provide clear, concise descriptions of purpose and functionality
- Document any non-obvious implementation details or design choices
- Include examples where helpful
- Explain parameters, return values, and potential exceptions

# Home folder CLAUDE.md

- @~/.claude/CLAUDE.md
