# IATA Principal Data Scientist Case Study

Project to analyse IATA Annual Reviews to identify trends and opportunities in the aviation industry using keyword analysis, unsupervised clustering, interactive data visualisation, and an assistant performing semantic search.

## Structure

- `data/`: Raw input and processed output
- `environment.yml`: Conda environment for reproducibility
- `models/`: Storage of BERTopic (keyword trends) and FAISS (assistant) models
- `notebooks/`: Interactive walkthrough with training and optimisation of keyword trend models and visualisations of results
- `outputs/`: Plots and other analysis outputs
- `src/`: Scripts for data extraction and semantic search assistant. Helper scripts in utils


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

## Download datasets and preprocess data

### Get data

Bash script to download all Annual Reviews and rename in a common format

```
./get_data.sh
```

### Preprocess data

Processes all pdf files in a given folder, to split them into chunks of max 1000 characters, following paragraph structure of original text and write out a json file (```data/chunks.json```) for further processing and analysis. 

```
python src/extract.py
```

## Unsupervised clustering analysis for keyword trends

Notebooks provided to do cluster analysis of keywords using unsupervised learning (BERTopic model), perform a grid search to optimise hyperparameters of the model, and visualisations of the results as a function of time.

#### Model training

```notebooks/clustering_analysis_training.ipynb```

Trains clustering models on ```data/chunks.json```, with options to filter by years (e.g. from 2015 onwards, or excluding pandemic years).

#### Model optimisation

```notebooks/grid_search_clustering_analysis.ipynb```

Grid search over UMAP and HDBScan parameters to optimise training parameters.  
* Initial coarse scan over min_cluster_size and n_topics. 
* Fine-tuned scan around promising min_cluster_size values and exploring min_dists settings.

#### Results visualisation

```notebooks/clustering_analysis_visualisation.ipynb```

Outputs:
* 2D intertopic distance map
* Top 10 trends over time
* Business opportunities table with ranked recommendations

## Aviation assistant

```src/assistant.py```

The assistant allows users to ask semantic questions about aviation industry trends, sustainability, passenger rights, cargo, business opportunities, and more.  
It retrieves the most relevant information from pre-embedded documents and summarises the results into a concise, human-readable answer.

Usage:

```python src/assistant.py```

Options:
| Flag | Description |
|:-----|:------------|
| `-v`, `--verbose` | Show top match distances and snippets |
| `--rebuild` | Force rebuild of FAISS index and document embeddings |

Examples:
```
python src/assistant.py -v
python src/assistant.py --rebuild
python src/assistant.py -v --rebuild
```

Example prompts provided in example_prompts.txt

#### How it works:
* Embeds document chunks using a SentenceTransformer model (all-MiniLM-L6-v2).
* Builds a FAISS index for efficient semantic similarity search.
* On user query:
   * Retrieves the top relevant documents.
   * Sorts results by most recent information first.
   * Summarises retrieved texts using a T5-small summariser.
* Supports forced rebuilds

## Simple keyword analysis

```src/simple_keyword_analysis.py``` or ```notebooks/simple_keyword_analysis.ipynb```

Simple keyword-based trend analysis and validation. Counts instances of user-defined keywords across documents and plots trends over time.

Usage:
```
python src/simple_keyword_analysis.py
```

## Future improvements to the assistant

* Dynamic distance-based decision on the number of documents to summarise instead of fixed number (TOP_K variable)
* Add metadata-based filtering (e.g., year, topic) to refine search results further
* Deploy as a web application with an interactive front-end to improve user experience
* Fine-tune the embedding model on aviation-specific documents for improved retrieval quality