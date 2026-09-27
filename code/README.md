# DASE7506 MP1 code submission

This directory is the cleaned final code release. Its top level contains exactly seven Python source files; the original contract test remains under `tests/`.

## Python files

- `common.py`, `evaluate.py`, `model.py`: original course runtime and evaluator.
- `student.py`, `train.py`: untouched original starter baseline.
- `student_v4.py`: final self-contained model used by the submitted checkpoint.
- `train_v2.py`: final training script; its default implementation is `student_v4`.

The final `student_v4.py` includes its backbone, continuous window cache, and compact trigram backoff in one file. It does not import any discarded intermediate student version.

## Other retained files

- `configs/`: the baseline and final experiment configurations.
- `data/`: the original tokenizer, benchmark text, and data manifest.
- `tests/`: the original course contract test.
- `requirements.txt`, `.gitignore`, and `RUN_LOG_TEMPLATE.csv`: original support files.
- `PACKAGE_MANIFEST.json`: SHA-256 inventory for this cleaned directory.

## Evaluate the final checkpoint

Extract the separately submitted checkpoint bundle, then run from this directory:

```powershell
conda activate dase7506-mp1
python evaluate.py --checkpoint path\to\checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```

The checkpoint records `implementation=student_v4`. Expected full-test BPB is `1.5082599771`; the checkpoint SHA-256 is `0f457cf814324e0b84ed9531eb1244ac0b764d3283f761dbdf48c3f9a27d7285`.

To train the final neural backbone with a configuration that does not enable attached trigram buffers:

```powershell
python train_v2.py --config configs\modern256x8.json --run-dir runs\new-run --device cuda --precision bf16
```
