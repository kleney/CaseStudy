# IATA Principal Data Scientist Case Study

Project to analyse IATA Annual Reviews to identify trends and opportunities in the aviation industry using keyword analysis and interactive data visualisation.

## Structure

- `notebooks/`: Interactive walkthrough with visualisations and commentary.
- `src/`: Reusable scripts for data extraction and analysis.
- `data/`: Raw input and processed output (safe to share).
- `environment.yml`: Conda environment for reproducibility.

## Quick start

### Environment setup

###### Initial setup

```
conda env create -f environment.yml
conda activate iata
python -m spacy download en_core_web_sm # download and install the small English language model "en_core_web_sm"
```

###### Subsequent times

```
conda activate iata
```

### Download datasets

Bash script to download all annual reviews and rename in a common format

```
./get_data.sh
```

### src/extract.py

Processes all pdf files in a given folder, to split them into chunks of max 1000 characters, following paragraph structure of original text and write out a json file (data/chunks.json) for further processing and analysis. 

Usage:
```
python src/extract.py
```

### src/simple_keyword_analysis.py

Takes the output of extract.py (data/chunks.json) to count instances of keywords from a user-defined list, prints this info in a simple table in the terminal, and makes a trend plot.

Usage:
```
python src/simple_keyword_analysis.py
```

Also available as an interactive notebook in notebooks/simple_keyword_analysis.ipynb

