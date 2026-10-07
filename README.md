# GLF-MDA

Code, input data, and final experimental outputs for Gated Low-rank Fusion for Microbe–Disease Association prediction (GLF-MDA).

## Contents

- Training and model code: train_mdmf.py, train_ablation.py, train_case_study_hmdad.py, model_mdmf.py, model_ablation.py, model.py, scalegcn.py, scale_gconv.py, types.py, and utils.py.
- Input data: the HMDAD/ and Disbiome/ folders contain the dataset files used by the experiments.
- Similarity and association matrices: the CSV files in the repository root provide the disease ontology, microbial functional similarity, and association matrices used by the two datasets.
- Final outputs: the Excel workbooks report dataset parameters and accuracy, formal ablation results, and the HMDAD case-study summary.

## Datasets

The experiments use the HMDAD and Disbiome microbe–disease association datasets. Public source data are included or identified by their original filenames. The similarity matrices are provided separately for HMDAD and Disbiome where applicable.

## Software environment

The implementation is written in Python and uses PyTorch, NumPy, pandas, scikit-learn, SciPy, and tqdm. A CUDA-enabled PyTorch installation can be used for GPU execution; the scripts also support CPU execution.

Install the main dependencies with:

```bash
pip install numpy pandas scipy scikit-learn tqdm openpyxl
# Install PyTorch from https://pytorch.org according to your operating system and CUDA version.
```

## Experiments

The main training script supports the HMDAD and Disbiome datasets, CVS1/CVS2/CVS3 evaluation settings, repeated five-fold evaluation, and the MDMF branch. Its principal options include --dataset, --cvs, --folds, --repeats, --epochs, --device, --alpha, and --beta.

The ablation script evaluates the component variants used in the manuscript, and train_case_study_hmdad.py runs the HMDAD case study.

The scripts were developed with the original project directory structure, in which the graph-convolution modules form a Python package named layers. When reproducing the experiments, preserve that package structure and set the data roots to the locations containing the HMDAD/ and Disbiome/ folders and the six similarity/association CSV files. Example commands are:

```bash
python train_mdmf.py --dataset HMDAD --cvs CVS1 --device cpu
python train_ablation.py --datasets HMDAD,Disbiome --cvs CVS1 --device cpu
python train_case_study_hmdad.py --dataset HMDAD --device cpu
```

Training can take substantial time. The final reported values are provided in the Excel workbooks in this repository.

## Reproducibility note

The uploaded files are the code, input data, and final experiment summaries associated with the manuscript. The result workbooks should be treated as the authoritative record of the reported experiments. Random seeds, fold settings, model dimensions, and other parameters are recorded in the scripts and summary workbooks.

## Contact

Weitong Sun  
Email: Sunweitong2004@outlook.com  
ORCID: https://orcid.org/0009-0003-9696-302X
