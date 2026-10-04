# ModelGuard

Security toolkit for ML models: **(1)** a static scanner for malicious pickle/PyTorch checkpoints, **(2)** signed model
manifests, **(3)** adversarial-robustness testing (FGSM/PGD + adversarial training), **(4)** backdoor detection.
NumPy for the model-side parts; the scanner and integrity checks are pure standard library.

```
python -m modelguard scan model.pt model.pkl      # exit 1 on critical findings
python -m modelguard sign ./model_dir --key $KEY  # writes MODEL_MANIFEST.json
python -m modelguard verify ./model_dir --key $KEY
```

## 1. Pickle scanner (most practically useful)
Loading a pickle can run arbitrary code. The scanner walks the opcode stream with `pickletools` **without executing it**,
resolves `GLOBAL`/`INST`/`STACK_GLOBAL` (including memoized strings), and flags dangerous callables (`os.system`,
`subprocess`, `eval`/`exec`, `socket`, `importlib`, ...) as **critical**, unknown classes as **medium**. Handles PyTorch zip
checkpoints (scans every `*.pkl` inside), `.npy` object arrays, truncated/corrupt streams (reported, no crash).
`safe_loads()` is a restricted unpickler that only permits allow-listed globals.

Tested: `os.system` payload detected in **all pickle protocols 0-5**; `eval`/`exec`/`Popen` detected; scanning never
runs the payload (marker-file test); real **scikit-learn and NumPy pickles produce no critical findings**; `safe_loads`
blocks the payload and still loads plain data. Limits: deny-list heuristic, so a novel gadget outside the list may only
show as *medium*; prefer `safetensors` for distributing weights.

## 2. Integrity manifests
SHA-256 per file + HMAC over the manifest. Detects modified (1 flipped bit), missing, and unexpected files, and a forged manifest.

## 3. Adversarial robustness (synthetic 64-feature, 5-class task, linear softmax model, clean accuracy 95.4%)

| L∞ budget ε (PGD) | 0 | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|
| standard training | 95.4% | 84.4% | 60.0% | 12.0% |
| adversarial training (ε=0.5) | 96.0% | 85.7% | 67.9% | 21.3% |

Adversarial training helps (+7.9 points at ε=0.5) but does not make the model robust; accuracy still collapses at ε=1.0.
PGD is at least as strong as FGSM (tested), and perturbations are verified to stay inside the ε-ball.
The model is linear, so gradients are exact; this demonstrates the method, it is not a benchmark on real image models.

## 4. Backdoor detection: what worked and what did not
A trigger (4 features set to 5.0 on 10% of training data, relabelled to class 0) gives **98% attack success** with ~99% clean accuracy.
- **Reverse-engineering the trigger (Neural-Cleanse style min-L1 search): did not work as a detector.** The backdoored class had the lowest required perturbation in 8/8 runs but never as a statistical outlier, so it flagged 0/8. It is kept in `detect_backdoor()` as a diagnostic only.
- **Weight-outlier check (`weight_outliers`, exact for linear models): partial.** Over 20 seeds there is no clean threshold:

| z threshold | backdoor found | clean model falsely flagged |
|---|---|---|
| 5 | 4/20 | 0/20 |
| **4 (default)** | **13/20** | **2/20** |
| 3 | 20/20 | 11/20 |

So it is a useful screening signal, not a verdict. It applies only to linear/softmax models.

## Development
```
pip install -e .[dev] && pytest -q     # 20 tests
```
MIT licensed.
