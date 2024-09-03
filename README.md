# Evaluation of impression-aware recommender systems (IARS)

This repository contains several utilities and scripts to perform experiments on IARS.

## Repository organization

Before executing any script, you create the respective environment and install its dependencies. Refer to the [Installation](#installation).

### Starting point

After you've created the environment and installed its dependencies, you may explore the scripts inside the [scripts](scripts) directory.
Inside that directory, we placed many scripts that perform different tasks, depending on your needs, each script is a potential starting point.
- For the evaluation and replication of several impression-aware recommenders, see [this section](#evaluation-of-impression-aware-recommender-systems-iars)
- For the evaluation of graph-based impression-aware recommenders, see [this section](#evaluation-of-graph-based-recommenders-with-impressions)
- To produce statistics of the different matrices used in the evaluations, e.g., UIM or URM, see [this section](#statistics-of-matrices-with-interactions-or-impressions)
- To produce statistics of the ContentWise Impressions dataset, see [this section](#statistics-of-the-contentwise-impressions-dataset)

#### Dataset Processing source code.
Three different files contain the code to load, process, compute features, and save to disk the datasets. 
- For the ContentWise Impressions dataset, see [ContentWiseImpressionsReader.py](impressions_evaluation/readers/ContentWiseImpressionsReader.py)
- For the MIND-SMALL and MIND-LARGE datasets, see [MINDReader.py](impressions_evaluation/readers/MINDReader.py)
- For the FINN.no Slates dataset, see [FINNNoReader.py](impressions_evaluation/readers/FINNNoReader.py)

#### Download all datasets

To run the experiments, you must download each dataset separately and place it in the corresponding folder of each dataset. The following map illustrates where to put the uncompressed files of each dataset. The scripts will do `a best effort attempt` to download the datasets, however, it may not be reliable due to network conditions. 
```
impressions-evaluation/
  |----> impressions_evaluation/
  |      |
  |----> data/
  |      |---->ContentWiseImpressions/
  |      |      |---->original/
  |      |      |      |  <Place dataset here> 
  |      |---->FINN-NO-SLATE/
  |      |      |---->original/
  |      |      |      |  <Place dataset here> 
  |      |---->MIND-SMALL/
  |      |      |---->original/
  |      |      |      |  <Place dataset here> 
  |      | ...
  | ...
```

### Evaluation and replication study of impression-aware recommenders

The [run_evaluation_study_impression_aware_recommenders.py](scripts/run_evaluation_study_impression_aware_recommenders.py) script is the main orchestrator to run the experiments on impression-aware recommenders from the literature. The script has the following console flags.
```python
class ConsoleArguments(Tap):
    create_datasets: bool = False
    """If the flag is included, then the script ensures that datasets exists, i.e., it downloads the datasets if possible and then processes the data to create the splits."""

    include_baselines: bool = False
    """Tunes the hyper-parameters of the base recommenders, e.g., ItemKNN, UserKNN, SLIM ElasticNet."""

    include_impressions_heuristics: bool = False
    """Tunes the hyper-parameter of time-aware impressions recommenders: Last Impressions, Recency, and Frequency & Recency. The first recommender does not need to be tuned, while the latter two must be tuned."""

    include_impressions_reranking: bool = False
    """Tunes the hyper-parameter of re-ranking impressions recommenders: HFC, Cycling, and IDF. These recommenders need base recommenders to be tuned, if they aren't then the method fails."""

    include_impressions_profile: bool = False
    """Tunes the hyper-parameter of the IUP recommenders. These recommenders need similarity-based recommenders to be tuned, if they aren't then the method fails."""

    print_evaluation_results: bool = False
    """Exports to Parquet, CSV, and LaTeX the accuracy, beyond-accuracy, optimal hyper-parameters, and scalability metrics of all tuned recommenders."""

    analyze_hyper_parameters: bool = False
    """Exports to Parquet, CSV, Tikz, PDF, and PNG the distribution of hyper-parameters and parallel plots of them."""
```

As an example, the following command runs the script with the option of tuning the hyper-parameters of baseline recommenders:
```shell
poetry run python scripts/run_evaluation_study_impression_aware_recommenders.py --include_baselines
```

#### Parallelize execution

This script uses [dask](https://dask.org) to parallelize the experiments using processes. We used AWS instances to run all the experiments. Depending on the number of available cores on your machine, you will need to adapt the default number of processes. 

By default, this repository uses `2` different processes, you can however, change this default by changing the `num_workers` key inside the [pyproject.toml](pyproject.toml) file. For example, if you want to use 4 processes, then change the key to `num_workers = 4`. If you want to disable parallelism, then set the key to `num_workers = 1`.

### Evaluation of graph-based recommenders with impressions

The [run_evaluation_study_graph_based_impression_aware_recommenders.py.py](scripts/run_evaluation_study_graph_based_impression_aware_recommenders.py) script is the main orchestrator to run the experiments on impression-aware recommenders from the literature. The script has the following console flags:
```python
class ConsoleArguments(Tap):
    create_datasets: bool = False
    """Ensures that datasets exists, i.e., it downloads the datasets if possible and then processes the data to create the splits."""

    include_baselines: bool = False
    """Tunes the hyper-parameters of the pure collaborative recommenders"""

    include_impressions: bool = False
    """Tunes the hyper-parameters of graph-based recommenders using impressions, i.e., the UIM."""

    include_impressions_frequency: bool = False
    """Tunes the hyper-parameters of graph-based recommenders using impressions frequency, i.e., the UIM-F"""

    print_evaluation_results: bool = False
    """Exports to Parquet, CSV, and LaTeX the accuracy, beyond-accuracy, optimal hyper-parameters, and scalability metrics of all tuned recommenders."""

    analyze_hyper_parameters: bool = False
    """Exports to Parquet, CSV, Tikz, PDF, and PNG the parallel plots of hyper-parameters."""
```

As an example, the following command runs the script with the option of tuning the hyper-parameters of graph-based impression-aware recommenders:
```shell
poetry run python scripts/run_evaluation_study_impression_aware_recommenders.py --include_impressions --include_impressions_frequency
```

### Statistics of matrices with interactions or impressions
The script [compute_evaluation_study_datasets_statistics.py](scripts/compute_evaluation_study_datasets_statistics.py) computes statistical properties of the matrices used in our evaluation studies, e.g., their density, number of non-zero elements, popularity among others. The script has the following console flags.
```python
class ConsoleArguments(Tap):
    create_datasets: bool = False
    """If the flag is included, then the script ensures that datasets exists, i.e., it downloads the datasets if possible and then processes the data to create the splits."""

    print_datasets_statistics: bool = False
    """Export to CSV statistics on the different sparse matrices existing for each dataset."""

    plot_datasets_popularity: bool = False
    """Creates plots depicting the popularity of each dataset split."""
```


### Statistics of the ContentWise Impressions dataset
The script [compute_statistics_thesis_cw_impressions_dataset.py](scripts/compute_statistics_thesis_cw_impressions_dataset.py) computes statistical properties of the ContentWise Impressions dataset, e.g., the skewness of impressions, interactions, among others. The script does not have console flags.

## Installation

This repository requires `Python 3.10` and uses `poetry` to manage the dependencies and build process of it. We tested this repository against [Linux](#linux-installation) and [macOS](#macos-installation). Currently, we do not support installations on Windows. Be aware that during the installation of dependencies you may see some warnings. The installation procedures for your OS will guide you through all the steps needed to execute our experiments.

### Linux Installation

- Enter the `impressions-evaluation` folder:
  ```bash
  cd impressions-evaluation/
  ```
- Install dependencies for `pyenv`, `poetry`, and the repo source code (this includes a C/C++ compiler).
  ```bash
  sudo apt-get update -y; sudo apt-get install gcc make python3-dev gifsicle build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev -y
  ```
- `Python 3.10.12` using [pyenv](https://github.com/pyenv/pyenv#installation)
  ```bash
  curl https://pyenv.run | bash
  ```
    - Remember to include `pyenv` in your shell
      [Section 2: Configure your shell's environment for Pyenv](https://github.com/pyenv/pyenv#basic-github-checkout).
    - Reload your shell (simple: quit and open again).
- `Poetry` using `pyenv`
   ```bash
   pyenv install 3.10.12
   pyenv local 3.10.12
   curl -sSL https://install.python-poetry.org | python3 -
   ```
    - Ensure to add `export PATH="/home/<your user>/.local/bin:$PATH"` to your bash profile (e.g., `~/.bashrc`
      , `~/.bash_profile`, etc)
- Download dependencies using `poetry`
  ```bash
  poetry env use 3.10.12
  poetry install
  poetry run pip install --no-use-pep517 lightfm
  ``` 

### macOS Installation

- Enter the `impressions-evaluation` folder:
  ```bash
  cd impressions-evaluation/
  ```
- `Command Line Tools for Xcode` from the [Apple's Developer website](https://developer.apple.com/download/more/?=xcode). These tools are required to have a `C` compiler installed in your Mac. You'll need a free Apple ID to access these resources.
  ```bash
  xcode-select --install
  ```
- `Homebrew` from [this page](https://brew.sh). `libomp` is needed for lightgbm.
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   brew update
   brew install openssl readline sqlite3 xz zlib hdf5 c-blosc libomp
   ```
- Add location of `hdf5` and `c-blosc` to your shell.
  ```bash
  export HDF5_DIR=/opt/homebrew/opt/hdf5
  export BLOSC_DIR=/opt/homebrew/opt/c-blosc
  ```
- `Python 3.10.12`
    - Using [pyenv](https://github.com/pyenv/pyenv#installation)
      ```bash
      curl https://pyenv.run | bash
      ```
- `Poetry` using `pyenv`
    ```bash
    pyenv install 3.10.12
    pyenv local 3.10.12
    curl -sSL https://install.python-poetry.org | python3 -
    ```
- Download dependencies using `poetry`
  ```bash
  poetry env use 3.10.12
  poetry install
  poetry run pip install --no-use-pep517 lightfm
  ```
