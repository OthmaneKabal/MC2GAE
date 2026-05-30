# Robustness of Graph Self-Supervised Learning to Real-World Noise

This repository contains the code and data associated with the paper:

**Robustness of Graph Self-Supervised Learning to Real-World Noise: A Case Study on Text-Driven Biomedical Graphs**

Paper: [https://arxiv.org/pdf/2605.05463](https://arxiv.org/pdf/2605.05463)

The repository provides the datasets and implementation needed to reproduce all results reported in the paper.

## 1. Prepare the Environment

Create and activate a fresh Python environment:

```bash
conda create -n mc2gae python=3.10
conda activate mc2gae
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

## 2. PLM-Only Baselines

To reproduce the PLM-only baseline results, use `src/model/evaluate_plm.py`.

Run the best PLM model:

```bash
python src/model/evaluate_plm.py data/UMLS/MM_mapped_nci_GS.xlsx sentence-transformers/all-MiniLM-L6-v2 --export-preds-path predictions_plm.xlsx
```

Run all PLM models listed in a text file:

```bash
python src/model/evaluate_plm.py data/UMLS/MM_mapped_nci_GS.xlsx --plm-models-file models.txt --summary-path results_summary.csv
```

Arguments:

- `data/UMLS/MM_mapped_nci_GS.xlsx`: path to the gold standard file. The file must contain at least the columns `term` and `label`.
- `sentence-transformers/all-MiniLM-L6-v2`: Hugging Face model name used as the PLM encoder.
- `--plm-models-file models.txt`: optional text file containing one PLM model name per line. Empty lines and lines starting with `#` are ignored.
- `--export-preds-path predictions_plm.xlsx`: optional output path for saving term-level predictions.
- `--summary-path results_summary.csv`: optional output path for saving the performance summary across all evaluated PLMs. Supported formats are `.csv`, `.xlsx`, and `.json`.

The script reports accuracy, macro-F1, macro-precision, and macro-recall.

## 3. Reproduce GSSL Methods

To reproduce the graph self-supervised learning methods, run:

```bash
python src/model/main.py
```

The experiments can be configured from:

```bash
src/model/config.py
```

In this configuration file, you can select the dataset, the GSSL methods to test, the encoder and decoder architectures, and the main training hyperparameters.

## 4. Prepare UMLS-NCI Data

If you have obtained the UMLS data after accepting the required license, prepare the UMLS-NCI data with:

```bash
python data/prepare_UMLS_NCI.py umls_folder
```

where `umls_folder` is the path to your local UMLS installation folder.

## Questions

If you have any questions, issues, or suggestions, please open an issue or contact us.

## Citation

If you use our data or methods, please cite:

```bibtex
@misc{kabal2026robustnessgraphselfsupervisedlearning,
      title={Robustness of Graph Self-Supervised Learning to Real-World Noise: A Case Study on Text-Driven Biomedical Graphs}, 
      author={Othmane Kabal and Mounira Harzallah and Fabrice Guillet and Hideaki Takeda and Ryutaro Ichise},
      year={2026},
      eprint={2605.05463},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.05463}, 
}
```
