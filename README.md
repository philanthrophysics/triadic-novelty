# Triadic Novelty Measures

This repository provides tools for analyzing scholarly publication data and computing advanced novelty metrics, including Pioneer, Maverick, and Vanguard novelty scores. The core logic is implemented in `triadic_novelty/measures.py`.

## Features

- **CitationData Class**: Main interface for analyzing citation data and computing novelty measures.
- **Novelty Metrics**:
  - **Pioneer Novelty**: Quantifies the introduction of new subject categories and their subsequent impact.
  - **Maverick Novelty**: Measures the novelty of subject category combinations and their influence over time.
  - **Vanguard Novelty**: Assesses the novelty and impact of subject category pairings using network-based approaches.
- **Base Model Generation**: Supports creation of null models with shuffled subject assignments for statistical comparison.
- **Flexible Analysis**: Allows customization of baseline and analysis periods, impact windows, and model parameters.

## Installation

To install the package, clone the repository and install the dependencies:

```bash
pip install pandas numpy python-igraph tqdm
```

## Usage

### 1. Prepare Your Data

Input data should be a pandas DataFrame with the following columns:
- `publicationID`: Unique identifier for each publication.
- `references`: List of referenced publication IDs.
- `subjects`: List of subject categories for each publication.
- `year`: Year of publication.

### 2. Initialize CitationData

```python
from triadic_novelty.measures import CitationData

citation_data = CitationData(
    data,  # your DataFrame
    baselineRange=(start_year, end_year),  # years for baseline period
    analysisRange=(start_year, end_year),  # years for analysis
    attractiveness=None,  # optional, for null model
    showProgress=True     # show progress bars
)
```

### 3. Calculate Novelty Scores

- **Pioneer Novelty:**
  ```python
  pioneer_scores = citation_data.calculatePioneerNoveltyScores(
      impactWindowSize=5,         # years after introduction to consider
      returnSubjectLevel=False    # set True for subject-level results
  )
  ```

- **Maverick Novelty:**
  ```python
  maverick_scores = citation_data.calculateMaverickNoveltyScores(
      backwardWindow=5,  # years before publication for baseline
      forwardWindow=5    # years after publication for impact
  )
  ```

- **Vanguard Novelty:**
  ```python
  vanguard_scores = citation_data.calculateVanguardNoveltyScores(
      weightsCount=4  # number of bins for edge weights
  )
  ```

### 4. Null Model Generation

To generate a base model with shuffled subject assignments:
```python
null_model = citation_data.generateBaseModelInstance(
    attractiveness=0.5,  # parameter for subject assignment
    showProgress=True
)
```

### Output

Each novelty calculation returns a pandas DataFrame with publication-level scores and relevant metrics. See the docstrings in `measures.py` for detailed descriptions of each method's output.

## Requirements

- Python 3.x
- pandas
- numpy
- igraph
- tqdm

### References

- For detailed methodology, see the docstrings in `triadic_novelty/measures.py`.
- Example usage can be found in `docs/example.py`.

## Contributing

We welcome contributions! Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature-branch`)
6. Create a new Pull Request
