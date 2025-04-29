# IATA Principal Data Scientist Case Study

Project to analyse IATA Annual Reviews to identify trends and opportunities in the aviation industry using keyword analysis, unsupervised clustering, interactive data visualisation, and an assistant performing semantic search.

## Structure

- `data/`: Raw input and processed output
- `environment.yml`: Conda environment for reproducibility
- `models/`: Storage of BERTopic (keyword trends) and FAISS (assistant) models
- `notebooks/`: Interactive walkthrough with training and optimisation of keyword trend models and visualisations of results
- `outputs/`: Plots and other analysis outputs
- `src/`: Scripts for data extraction and RAG assistant. Helper scripts in utils


## Quick start

### Environment setup

###### Initial setup

```
conda env create -f environment.yml
conda activate iata
python -m spacy download en_core_web_sm # download and install the small English language model "en_core_web_sm"
```

<small>*N.B. ```environment.yml``` captures the working state as of April 2025 and was updated in reponse to VSCode updates that broke dependencies. It includes all required dependencies for the project but has not been cleaned. Some platform-specific packages may be present but they do not affect cross-platform compatibility. The previous environment file ```old_environment.yml``` is included for completeness.*</small>

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

The assistant allows users to ask questions about aviation industry trends, sustainability, passenger rights, cargo, business opportunities, and more. It retrieves the most relevant information from pre-embedded documents and supports answering questions based on semantic search and Retrieval-Augmented Generation (RAG).

- **Semantic Retrieval:** Queries are embedded and matched against aviation industry reports using FAISS.
- **RAG Generation:** Retrieved contexts are sent to an LLM (OpenAI gpt-3.5-turbo) to generate answers.

The mode is set via the ```RAG_MODE``` variable in ```src/assistant_config.py```

#### RAG setup prerequisites

To use the RAG functionality, you must have an OpenAI API key and active paid credits for OpenAI API access.

**OpenAI API key**
1. Generate your API key from [OpenAI Platform](https://platform.openai.com/account/api-keys).
2. Set it as an environment variable:

```bash
export OPENAI_API_KEY="your-secret-key"
```
Optionally, add it permanently to your ```~/.bashrc``` or ```~/.zshrc```.

**API credit**
1. Go to the [OpenAI billing page](https://platform.openai.com/account/billing/overview) and give Sam Altman some of your hard-earned cash.

#### Configuration

All configurables, folder paths, model choices, and whether to run in RAG or retrieval-only mode, are set in src/assistant_config.py.  

#### Usage:

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

Example prompts are provided in ```example_prompts.txt```

#### How it works:
The workflow supports two operational modes:

#### 1. Retrieval-Only Mode

- **Query embedding:**  The user's question is embedded using a SentenceTransformer model (all-MiniLM-L6-v2).
- **Semantic search:**  FAISS is used to find the most relevant text chunks from the indexed documents.
- **Summarisation:**  Retrieved chunks are summarised using a lightweight text summarisation model (t5-base) to generate an answer.

**Use case:**  When no OpenAI API key is available, or when a pure retrieval-based summary is preferred.

#### 2. Retrieval-Augmented Generation (RAG) Mode

- **Query embedding and retrieval:** Same as in retrieval-only mode — semantic search finds the most relevant text chunks.
- **Prompt building:** The retrieved chunks and the user’s question are combined into a prompt.
- **OpenAI API call:** The prompt is sent to an LLM (GPT-3.5 Turbo) via OpenAI’s API to generate a context-aware answer.
- **Fallback handling:** If the OpenAI API key is missing the assistant automatically switches back to retrieval-only mode without crashing.

#### Future improvements to the assistant

- Develop a lightweight web UI
- Enhance retrieval with cross-encoder re-ranking.
- Enable multi-hop reasoning for complex queries.
- Add metadata-based filtering (e.g., year, topic) to refine search results further
- Fine-tune the embedding model on aviation-specific documents for improved retrieval quality
- Include citations in generated answers.


## Extra: Simple keyword analysis

```src/simple_keyword_analysis.py``` or ```notebooks/simple_keyword_analysis.ipynb```

Simple keyword-based trend analysis and validation. Counts instances of user-defined keywords across documents and plots trends over time.

Usage:
```
python src/simple_keyword_analysis.py
```