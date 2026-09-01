# P2SDiff v2 reproducibility bundle

This bundle records the exact data order and configurations used for the
reported v2 experiments. It does not contain the medical images.

## Dataset splits

`splits/<dataset>_{train,val,test}.txt` contains the image IDs in each split.
`splits/<dataset>.csv` additionally records the original array index and the
relative image/mask paths.

The order matters. The original preprocessing used unsorted `glob.glob()`
output and then split the resulting arrays by fixed index ranges. Recreating
the arrays with a different file order produces different partitions.

Counts:

- PH2: 80 train / 20 validation / 100 test
- ISIC2017: 1250 train / 150 validation / 600 test
- ISIC2018: 1815 train / 259 validation / 520 test
- HAM10000: 7200 train / 1800 validation / 1015 test

## Verification

`array_metadata.json` contains the exact shape, dtype, value range, file size,
and SHA-256 checksum of each reference `X_` and `Y_` npy file. Matching hashes
mean that preprocessing and ordering are identical. Different hashes do not
necessarily mean the source images differ; a different array order also
changes the hash.

## Reference runs

`reference_runs/` contains the exact `config.json`, `summary.json`, and
`test_metrics.json` files from the reported runs. Local paths such as
`skin_root`, `out_dir`, and `init_checkpoint` must be adapted to the target
machine. Other parameters should match for replication.

`reference_environment.json` records the source commit and key package
versions. Exact GPU kernels can still introduce small numerical differences,
but they should not explain large Dice-score gaps.

For `ft_ham` runs, initialize from the `best.pth` checkpoint of the selected
HAM10000 reference run. Evaluation should use `best.pth`; the guidance scale
is read from the saved configuration and is 1.5 for these v2 runs.
