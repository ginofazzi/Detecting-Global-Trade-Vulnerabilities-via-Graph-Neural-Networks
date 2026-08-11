# Extra baselines: methodology and evaluation

This note describes the two graph-native, non-learned baselines implemented in `code/extra_baselines.py`. Both assign one continuous risk score to each country (node) from the weighted trade graph; a country is classified as at risk when its score is at least a threshold selected on the training data.

## Baselines

For a directed edge from exporter/supplier `i` to importer `j`, let `w_ij` be its (non-negative) trade value and let `s_ij = w_ij / sum_k w_kj` be supplier `i`'s share of importer `j`'s total imports.

- **Inverse partner out-degree (weighted out-strength).** Supplier `i` has weighted out-strength `d_i^out = sum_j w_ij`. The importer-level score is
  `risk_j = sum_i s_ij / (d_i^out + eps)`.
  It is high when an importer relies on suppliers with low total export strength, with the contribution of each supplier weighted by its import share.

- **Import concentration (HHI).** The score is
  `HHI_j = sum_i s_ij^2`.
  It is one when all imports come from a single supplier and decreases as sourcing becomes more diversified.

Nodes without incoming imports receive a score of zero. When graph edge values are stored on the log scale, they are exponentiated before either score is calculated, so that the metrics use trade-value weights.

## Procedure

1. For each graph, compute one risk score per node and retain its graph metadata (`digits`, year, graph type, commodity, and country ID) together with the observed binary label `y`.
2. Use all 2012–2020 graphs as the training set and hold out the 2021 graphs as the test set. For multilayer graphs, scores are generated separately for every product-year graph; for multigraphs, there is one all-products graph per year.
3. For each baseline independently, choose a classification threshold using training scores only. The candidate set consists of the unique values from 500 equally spaced empirical quantiles of the finite training scores, plus `-infinity` and `+infinity`.
4. Classify a node as positive when `risk >= threshold`. Select the threshold that maximizes positive-class F1; ties are resolved by higher recall and then higher precision.
5. Evaluate the selected, fixed threshold on both the training set and the untouched 2021 test set. Report positive-class F1, precision, recall, accuracy, the confusion-matrix counts (TN, FP, FN, TP), sample count, observed positive rate, and predicted-positive rate.

The implementation and runnable evaluation workflow are in `code/extra_baselines.py` and `code/extra_baselines.ipynb`, respectively.
