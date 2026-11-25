# Residual Topic Model: Interpretable Ordered Topics and the Choice of the Number of Topics

Code, prepared datasets, and results accompanying the paper:
*Koltcov S., Ignatenko V., Surkov A. “Residual Topic Model: Interpretable Ordered Topics and the Choice of the Number of Topics”* (Laboratory for Social and Cognitive Informatics, HSE University,
St. Petersburg).

This repository contains the Residual Topic Model (RTM), the baselines (Gibbs LDA, ProdLDA, and ETM),
three criteria for selecting the number of topics (the energy knee, the minimum median Jensen–Shannon divergence
between runs, and the maximum top-word silhouette), eleven prepared corpora, and criterion curves for all trained
models. These results allow every table in the paper to be reproduced without retraining
(`notebooks/05_paper_tables.ipynb`).

## Repository structure

```text
rtm/                  Python package
  model.py            RTM: encoder, codebook, matching pursuit, topics as deviations from the background (Section 3)
  train.py            RTM training (Algorithms 2–3), reproducible for a fixed seed
  baselines.py        ProdLDA and ETM
  lda.py, _glda_job.py LDA with Gibbs sampling (the laboratory's sampler, Windows)
  synth.py            Synthetic corpus generator
  criteria.py         Energy and its knee, median JS divergence between runs, silhouette, topic recovery
  scan.py             Directory layout for model sweeps and computation of criterion curves
  summary.py          Topic-count estimates, mean errors, exact matches, sign test (Tables 11–20)
  data.py             Corpus registry, T grids, data loading
notebooks/
  01_datasets.ipynb       Corpora (Table 4), regeneration of synthetic data
  02_rtm.ipynb            RTM training, topics, document topics, sweep over T
  03_baselines.ipynb      LDA, ProdLDA, ETM
  04_criteria.ipynb       Three criteria computed from trained models -> results/*.csv
  05_paper_tables.ipynb   Tables 5, 8, 10–20 and Figures 8–9 of the paper
data/<corpus>/        Prepared corpora (see below)
results/              Criterion curves for all models in the paper
  curves.csv          Corpus, model, run, T, energy E, silhouette
  stability.csv       Corpus, model, representation (full | lift), T, median JS divergence (bits)
  recovery.csv        Synthetic data: number of recovered ground-truth topics
third_party/pyGibbsLDA2/  The laboratory's Gibbs sampler (gibbsldacore.dll and Python wrapper)
```

## Installation

```bash
git clone https://github.com/hse-scila/RTM.git
cd RTM
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks
```

For a CUDA-enabled PyTorch installation, follow the instructions at [pytorch.org](https://pytorch.org).
The paper used PyTorch 2.14.1 with CUDA 12.6. All components also run on CPU, but training RTM and the neural
baselines takes several times longer.

## Using the notebooks

Run the notebooks in order from the `notebooks/` directory. Each notebook can also be run independently.

1. **Verify the paper's tables only:** run `05_paper_tables.ipynb`. It reads `results/*.csv` and, within a few
   seconds, produces topic-count estimates for the three criteria, mean errors, exact-match counts, sign-test
   results, JS minima, energy knees, topic-recovery results, and plots of the criterion curves.
2. **Explore the data:** run `01_datasets.ipynb` for a corpus summary and regeneration of the synthetic corpora
   from the ground-truth topics. The regenerated data match the supplied data element for element.
3. **Train a model:** use `02_rtm.ipynb` for RTM (Lenta, T = 10: approximately 4 minutes on GPU for 450 epochs;
   set `EPOCHS = 20` for a quick trial) and `03_baselines.ipynb` for LDA, ProdLDA, and ETM. Models are saved to
   `models/<model>/<corpus>/s<run>/T<T>/`.
4. **Repeat the full sweep:** set `RUN_SCAN = True` in notebooks 02 and 03 (approximately 600 RTM models,
   each taking 4–30 minutes on GPU, and the same number of models for each of the three baselines).
   Then set `RUN = True` in `04_criteria.ipynb` to recompute the curves in `results/recomputed/`.
   To generate tables from these curves, set `RES = ROOT / "results" / "recomputed"` in
   `05_paper_tables.ipynb`.

The topic-count sweep uses T = 4…30 in steps of 2, with steps of 1 in the window K − 2…K + 2, and three runs
for each T (`rtm.data.grid`). For the synthetic corpora with 6 and 20 topics, the grids follow Table 2 of the
paper. RTM is trained for 400 epochs on synthetic corpora and 450 epochs on real corpora. An RTM run with a final
loss greater than 100 is considered invalid (one run on 20 Newsgroups at T = 4 in the paper).

## Data

| Corpus | Language | Documents | Vocabulary V | Categories / K |
|---|---|---:|---:|---:|
| `synth_K6`, `synth_k`, `synth_K20`, `synth_K12_uneq` | — | 6,000 | 11,229 | 6, 12, 20, 12 |
| `lenta_E2` (Lenta) | Russian | 8,650 | 11,229 | 10 |
| `news20_clean` (20 Newsgroups) | English | 15,720 | 15,347 | 20 |
| `mnds` (MN-DS) | English | 10,829 | 21,701 | 17 |
| `gnad10` (GNAD10) | German | 10,252 | 27,663 | 9 |
| `sanad` (SANAD) | Arabic | 10,470 | 9,830 | 7 |
| `thucnews` (THUCNews) | Chinese | 10,460 | 30,000 | 14 |
| `livedoor` (Livedoor) | Japanese | 7,365 | 17,101 | 9 |

Each corpus contains `bow.npz` (a bag-of-words matrix in SciPy CSR format), `vocab.txt`, `stats.json`,
`docs.txt.gz` (lemmatized documents with stop words removed, used as input to the Gibbs sampler),
`w2v_vectors.npy` (the word2vec vectors used to compute the silhouette in the paper), and `labels.npy`
(document categories, where available). Synthetic corpora additionally include `phi_true.npy`, `truth.csv`,
and `popularity.npy`.

Vocabulary size is capped at 30,000 words. Words occurring in fewer than five documents are removed from all
corpora except Lenta. The 20 Newsgroups corpus is built from the `sklearn` 20news-bydate dataset
(train + test), with headers, signatures, and quoted text removed. Documents shorter than ten tokens are discarded.

The original collections belong to their respective authors. When using the data, cite the original corpus
publications listed in the paper.

## Reproducibility

* With the same seed and device, RTM is exactly reproducible: the Lenta model with T = 10 and seed 0 trained
  using `rtm.train.train_rtm` matches the model used in the paper (maximum difference between topic
  distributions: 0). Results may differ slightly on another GPU, with another PyTorch version, or on CPU.
* ProdLDA and ETM are reproducible under the same conditions (the Lenta models with T = 10 and seed 0 match
  those used in the paper).
* The `results/*.csv` files were computed using the package functions (`rtm.scan`) for all 2,243 models
  of the four types studied in the paper. The topic-count estimates produced from these files by
  `05_paper_tables.ipynb` match the paper's tables.
* The LDA Gibbs sampler uses its own random initialization. Repeated runs produce different topics, and
  criterion-based estimates may shift to neighboring values of T.
* Multithreaded word2vec training is not bitwise reproducible, so the vectors used in the paper are saved in
  `data/<corpus>/w2v_vectors.npy`.
* `gibbsldacore.dll` works only on Windows. On other systems, LDA can be trained using any implementation
  of collapsed Gibbs sampling (200 iterations, α = β = 0.01). Save the topics to `phi200.csv` in the same
  format (`word;p(w|t_1);…;p(w|t_T)`), and notebooks 04–05 will work without changes.

## License

The code in `rtm/` and the notebooks are distributed under the MIT License (see `LICENSE`).
This license does not cover the original corpus texts in `data/`, whose rights belong to the corpus authors,
or the sampler in `third_party/pyGibbsLDA2`. The latter is the Gibbs sampler developed by the Laboratory for
Social and Cognitive Informatics at HSE University and is included to reproduce the LDA results.

## Citation

If you use the code or data, please cite the paper. Full bibliographic details will be added after publication.
