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
- [ ] Wrapper class validation passes with Time-MMD dataset
- [ ] Forward pass produces expected output shapes and ranges
- [ ] Training loop integrates smoothly with wrapper class
- [ ] Model can process both time series and text inputs
- [ ] Training pipeline completes without errors
- [ ] Ablation study shows meaningful performance comparison
- [ ] Results demonstrate the value (or lack thereof) of text information
- [ ] Code is well-documented and reproducible

## Next Steps

1. - [x] Initialize the uv project and set up basic structure
2. - [x] Install TimesFM and verify compatibility
3. - [x] Implement basic wrapper class skeleton
4. - [ ] Set up Time-MMD dataset loading and preprocessing
   - [ ] Implement Time-MMD dataset loader
   - [ ] Create data preprocessing utilities
5. - [ ] Validate wrapper class functionality with Time-MMD data
   - [ ] Test basic forward pass with time series data only
   - [ ] Verify input/output shapes and data flow
   - [ ] Ensure wrapper properly delegates to underlying TimesFM
   - [ ] Run sanity checks with small subset of Time-MMD dataset
   - [ ] Confirm training loop works with wrapper class
6. - [ ] Begin model architecture enhancements

## Notes

- Consider using mixed precision training for efficiency
- Implement proper error handling for multimodal input validation
- Document any architectural changes made to accommodate multimodal inputs
- Keep detailed logs of hyperparameter choices and their impact

## Bash Commands

- `uv run mypy .`: Type checking
- `uv run ruff check`: Linting
- `uv run ruff format`: Code formatting

# Home folder CLAUDE.md

- @~/.claude/CLAUDE.md
