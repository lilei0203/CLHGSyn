
Output Files
- results/<dataset>/summary.json: Five-fold means, sample standard deviations (ddof=1), pooled out-of-fold (OOF) metrics, and independent test metrics computed from the average predicted probabilities of the five models.
- fold_metrics.csv: Per-fold metrics, training/validation/test sample counts, and the best epoch.
- oof_predictions.csv: Outer test-fold predictions, with exactly one prediction for each sample in the development set.
- fold_1 through fold_5: best.pt, history.csv, split_indices.npz, training_positive_hyperedges.npy, and predictions for the outer test fold and independent test set.
- config.json, node_mapping.json, and protein_neighbors.npz: Run configuration, node mappings, and fixed protein neighborhoods required to restore the model.
- samples.csv and data_report.json: Original row indices, labels, group IDs, dataset statistics, and SHA-256 hashes of the source files.
- results_training.log: Training log for the main experiment. Results from the two-epoch smoke tests are excluded from the reported results.
ACC, Precision, Recall, and F1 are calculated using a fixed probability threshold of 0.5. AUPR is computed using trapezoidal integration of the precision–recall curve. AP (average precision) is reported separately to distinguish the two definitions. Pooled OOF metrics and mean per-fold metrics are calculated and stored separately.

Verification and Standalone Prediction
python -m unittest discover -s tests -v
python verify_results.py --results results
python summarize_results.py
python predict.py --result-dir results/OncologyScreen --fold 1 --input examples/triplets.csv --output predictions.csv
The prediction input must contain the columns drug_a, drug_b, and cell. Drugs must be specified using DrugBank IDs, and cell line names must match those in the dataset. Only load checkpoints trained by this project or obtained from trusted sources. Training, prediction, and verification commands can each be run independently.
