<h1 align="center">THC</h1>
<h2 align="center">Tweet Harmfulness Classification</h2>

<p align="center">
    <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3.7%20%7C%203.8-blue" alt="Python"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
    <a href="https://gitmoji.carloscuesta.me"><img src="https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67.svg?style=flat-square" alt="Gitmoji"></a>
</p>

This repository provides solution to **Task 6-2: Type of harmfulness** of
[PolEval 2019](http://2019.poleval.pl/) challenge. The establishment of this
project was guided by one simple mission:

<!-- prettier-ignore -->
> To create a world, where _haters ain't gonna hate_.

```
(0) RT @anonymized_account @anonymized_account wszystkiego co najlepsze i najpiękniejsze! 🎉💝
(1) @anonymized_account A ja bym to tak ujął: Kto krzyżem wojuje, na krzyżu ginie😁
(2) @anonymized_account @anonymized_account @anonymized_account Sakiewicz, Tobie wazelina oczy zalewa i bredzisz.
```

## Project Organisation

    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── logs               <- Tensorboard model training logs.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering) and
    │                         a short `-` delimited description, e.g. `00-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    |
    ├── poetry.lock        <- File to resolve and install all dependencies listed in the
    │                         pyproject.toml file.
    ├── pyproject.toml     <- File orchestrating the project and its dependencies.
    │
    ├── thc                <- Source code for use in this project.

### Notebooks

The project is designed to separate the particular modeling steps into
notebooks. Notebook list:

- [00-texts-integrity](https://github.com/mrtovsky/thc/blob/main/notebooks/00-texts-integrity.ipynb)
  focuses on getting familiarity with data and examines dataset imbalance. It
  also generates a presentation of an example input.
- [01-train-valid-split](https://github.com/mrtovsky/thc/blob/main/notebooks/01-train-valid-split.ipynb)
  is dedicated to dividing the data set into an appropriately represented
  training and validation set to avoid consequences of _sampling bias_ like
  shown in the widely known _The Literary Digest_
  [Presidential poll](https://en.wikipedia.org/wiki/The_Literary_Digest#Presidential_poll).
- [10-distilbert]() provides [DistilBERT](https://arxiv.org/abs/1910.01108)
  experiments setup. The **Multilingual Cased DistilBERT** model was fine-tuned
  on a downstream task trained with the use of an
  [AdamW](https://www.fast.ai/2018/07/02/adam-weight-decay/) optimizer.
- [11-model-selection]() shows method of selecting the best model with use of
  the **TensorBoard** training logs and prepares test dataset predictions.

## Installation

If only the **thc** source package functionalities are of interest then it is
enough to run:

```bash
pip install git+https://github.com/mrtovsky/thc.git
```

To interact with the notebooks e.g. rerun them, full project preparation is
necessary. It can be done in the following few steps. First of all, you need to
clone the repository:

```bash
git clone https://github.com/mrtovsky/thc.git
```

Then, enter this directory and create a **.env** file that stores environment
variable with the cloned repository path:

```bash
cd footvid/
touch .env
printf "REPOSITORY_PATH=\"$(pwd)\"" >> .env
```

### Poetry

The recommended way of installing the full project is via
[Poetry](https://python-poetry.org/docs/#:~:text=Linux%20and%20OSX.-,Installation,recommended%20way%20of%20installing%20poetry%20.)
package. If Poetry is not installed already, follow the installation
instructions at the provided link. Then, assuming you have already entered the
**thc** directory, resolve and install dependencies using:

```bash
poetry install
```

Furthermore, you may want to attach a kernel with the already created virtual
environment to Jupyter Notebook. This can be done by calling:

```bash
poetry run python -m ipykernel install --name=thc-venv
```

This will make **thc-venv** available in your Jupyter Notebook kernels.

### pip

It is also possible to install the package in a traditional way, simply run:

```bash
pip install -e .
```

This will install the package in an editable mode. If you installed it inside
of the virtual environment, then attaching it to the Jupyter Notebook kernel is
the same as with the **Poetry** but the command is stripped from the first two
elements (remember that the virtualenv needs to be activated beforehand):

```bash
python -m ipykernel install --name=thc-venv
```

## Results

| Dataset |    Micro-F1 |    Macro-F1 |
| ------- | ----------: | ----------: |
| TRAIN   | PLACEHOLDER | PLACEHOLDER |
| VALID   | PLACEHOLDER | PLACEHOLDER |
| TEST    | PLACEHOLDER | PLACEHOLDER |

More detailed training results can be displayed by opening the **TensorBoard**:

```bash
tensorboard --logdir ./logs/ --host localhost
```
