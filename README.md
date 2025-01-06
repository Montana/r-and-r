# R&R From Casetext Improved 

This project provides tools for handling datasets, querying documents, and analyzing results using different models like OpenAI and Anthropic.

## Features

- **Dataset Management**: Load and preprocess datasets like NQ, SQuAD, HotPotQA, and PubMed.
- **Query Handling**: Generate prompts, split documents, and retrieve answers.
- **Abbreviation and Chunking**: Efficiently process large documents by chunking and abbreviation.
- **Result Analysis**: Generate tables and results for various configurations.

## Requirements

- Python 3.8 or later
- OpenAI GPT-4 or Anthropic Claude API keys
- Dataset files placed in the `data` directory

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment files:
   - Create a `configs` directory.
   - Add `.env` files for OpenAI and Anthropic with the following variables:

#### `configs/openai.env`
   ```env
   OPENAI_KEY=your_openai_api_key
   OPENAI_ORG_ID=your_openai_org_id
   MODEL_NAME=gpt-4
   ```

#### `configs/anthropic.env`
   ```env
   ANTHROPIC_API_KEY=your_anthropic_api_key
   MODEL_NAME=claude-v1
   ```

4. Download datasets and place them in the `data` directory.

## Usage

### Run Dataset Analysis
To execute the dataset analysis, run:
```bash
python analysis.py
```

### Compare Baseline vs. Reprompt

```bash
python baseline_vs_reprompt.py
```

### Build PubMed Dataset

```bash
python build_pubmed.py
```

### Generate Result Tables

```bash
python tables.py
```

### Custom Runs

To run custom configurations, edit the parameters in the corresponding scripts (`run.py`, `baseline_vs_reprompt.py`, etc.).

## Folder Structure

```
.
├── analysis.py
├── baseline_vs_reprompt.py
├── build_pubmed.py
├── configs/
│   ├── openai.env
│   └── anthropic.env
├── data/
│   ├── nq/
│   ├── squad/
│   ├── hotpotqa/
│   └── pubmed/
├── models.py
├── parsers.py
├── prompts.py
├── run.py
├── tables.py
└── requirements.txt
```

## Author

Michael Mendy (c) 2025. 
