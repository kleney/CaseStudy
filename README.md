# IATA Principal Data Scientist Case Study

Project to analyse IATA Annual Reviews to identify trends and opportunities in the aviation industry using keyword analysis and interactive data visualisation.

## Structure

- `notebooks/`: Interactive walkthrough with visualisations
- `src/`: Reusable scripts for data extraction and helper scripts
- `data/`: Raw input and processed output
- `environment.yml`: Conda environment for reproducibility

## Quick start

### Environment setup

###### Initial setup

```
conda env create -f environment.yml
conda activate iata
python -m spacy download en_core_web_sm # download and install the small English language model "en_core_web_sm"
mkdir data output models
```

###### Subsequent times

```
conda activate iata
```

## Download datasets and preprocess data

### Get data

Bash script to download all annual reviews and rename in a common format

```
./get_data.sh
```

### Preprocess data

Processes all pdf files in a given folder, to split them into chunks of max 1000 characters, following paragraph structure of original text and write out a json file (data/chunks.json) for further processing and analysis. 

Usage:
```
python src/extract.py
```

## Analysis

### Simple keyword analysis

```src/simple_keyword_analysis.py``` or ```notebooks/simple_keyword_analysis.ipynb```

Takes the output of extract.py (data/chunks.json) to count instances of keywords from a user-defined list, prints this info in a simple table in the terminal, and makes a trend plot.

Usage:
```
python src/simple_keyword_analysis.py
```

### Unsupervised clustering analysis

Notebooks provided to do cluster analysis of keywords using unsupervised learning (BERTopic model), perform a grid search to optimise hyperparameters of the model, and visualisations of the results.

#### Model training

```notebooks/clustering_analysis_training.ipynb```

Trains models using the output of extract.py (data/chunks.json).  Can apply filtering of years to consider in the training model, e.g. since 2015, or excluding pandemic years.

#### Model optimisation

```notebooks/grid_search_clustering_analysis.ipynb```

Grid search over UMAP and HDBScan parameters to optimise training parameters.  Initial coarse scan over min_cluster_size and n_topics. Second, more fine-tuned scan around promising min_cluster_size values and exploring min_dists settings.

#### Results visualisation

```notebooks/clustering_analysis_visualisation.ipynb```

Outputs the diagnostic 2D intertopic distance map, a plot of top 10 trends over time, and a business opportunities table with ranked, prioritised recommendations.