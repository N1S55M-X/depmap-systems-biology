# DepMap Systems Biology

## Overview

This project explores relationships between gene expression programs and CRISPR dependency programs using DepMap datasets.

Instead of analyzing genes individually, genes are grouped into modules using KMeans clustering. The project then investigates how expression modules relate to dependency modules using correlation analysis and XGBoost.

The goal is to discover systems-level biological patterns and generate testable hypotheses.

---

## Dataset

This project uses publicly available DepMap datasets.

Download from:

https://depmap.org/portal/download/

Required files:

- Expression (Short-read) Public 26Q1
- CRISPR Gene Dependency Public 26Q1

After downloading, place the files in the same directory as `main.py`.

The datasets are not included in this repository.

---

## Method

### Data Processing

- Match common DepMap cell lines
- Remove CRISPR genes with >20% missing values
- Median imputation
- StandardScaler normalization

### Module Discovery

Genes are grouped into 50 expression modules and 50 CRISPR dependency modules using KMeans clustering.

### Module Relationships

Correlations are computed between every expression module and every CRISPR dependency module.

### Prediction

XGBoost models are trained to predict CRISPR dependency modules from expression modules.

---

## Results

Top predictive dependency modules:

<img width="380" height="272" alt="image" src="https://github.com/user-attachments/assets/679042ec-a326-4880-adc8-2ad2263fdfdb" />


These results suggest that some dependency programs can be partially predicted from coordinated expression programs.

---

## Requirements

Install:

```bash
pip install pandas numpy scikit-learn xgboost
```

---

## Run

```bash
python main.py
```

---

## Scientific Interpretation

This project demonstrates that:

- Gene dependencies are often organized as networks rather than isolated genes.
- Expression programs can predict certain dependency programs.
- Module-level analysis captures stronger biological signals than many single-gene analyses.

The results should be considered exploratory and hypothesis-generating.

---

## Limitations

- KMeans modules are mathematical clusters, not validated pathways.
- Results are based only on DepMap cell lines.
- Independent biological validation is required.
- Findings are not intended for clinical use.
 
---
## Note

This project is exploratory and hypothesis-generating.

Gene modules are generated before train/test splitting, which may inflate predictive performance estimates. The reported R² values should therefore be interpreted as exploratory rather than definitive predictive benchmarks.

---

## Future Work

Potential improvements include:

- WGCNA
- NMF
- Pathway enrichment analysis
- Graph neural networks
- Drug sensitivity integration
- Multi-omics analysis

---
