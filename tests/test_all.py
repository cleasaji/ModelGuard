import os
import pickle
import subprocess
import sys
import zipfile
import numpy as np
import pytest
from modelguard import *
from modelguard.adversarial import SoftmaxModel


class Evil:
    def __init__(self, fn, arg):
        self.fn, self.arg = fn, arg

    def __reduce__(self):
        return (self.fn, (self.arg,))


@pytest.mark.parametrize("proto", range(0, 6))
def test_os_system_detected_in_every_protocol(proto):
    data = pickle.dumps(Evil(os.system, "echo pwned"), protocol=proto)
    f = scan_bytes(data)
    assert any(x.severity == "critical" and "system" in x.callable for x in f), (proto, f)


@pytest.mark.parametrize("fn,arg,needle", [(eval, "1+1", "eval"), (subprocess.Popen, "ls", "Popen"),
                                           (exec, "x=1", "exec")])
def test_other_dangerous_callables(fn, arg, needle):
    f = scan_bytes(pickle.dumps(Evil(fn, arg), protocol=4))
    assert any(x.severity == "critical" and needle in x.callable for x in f)


def test_scanning_never_executes_payload(tmp_path):
    marker = tmp_path / "pwned"
    data = pickle.dumps(Evil(os.system, f"touch {marker}"), protocol=2)
    scan_bytes(data)
    assert not marker.exists()


def test_benign_sklearn_and_numpy_pickles_not_flagged():
    from sklearn.linear_model import LogisticRegression
    X, y = make_dataset(200, 8, 3)
    for obj in (LogisticRegression(max_iter=200).fit(X, y), {"w": np.arange(10.0), "names": ["a", "b"]}):
        assert [x for x in scan_bytes(pickle.dumps(obj)) if x.severity == "critical"] == []


def test_zip_checkpoint_scanning(tmp_path):
    p = tmp_path / "model.pt"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("archive/data.pkl", pickle.dumps(Evil(os.system, "x"), protocol=2))
        z.writestr("archive/data/0", b"\0" * 16)
    f = scan_file(str(p))
    assert f and f[0].member == "archive/data.pkl" and f[0].severity == "critical"


def test_corrupt_stream_reported_not_crashed():
    assert scan_bytes(b"\x80\x04\x95garbage")[0].callable == "<parse-error>"


def test_safe_loads_blocks_payload_allows_plain_data(tmp_path):
    marker = tmp_path / "pwned"
    with pytest.raises(pickle.UnpicklingError):
        safe_loads(pickle.dumps(Evil(os.system, f"touch {marker}")))
    assert not marker.exists()
    assert safe_loads(pickle.dumps({"a": [1, 2, {3}], "b": (1.5, "x")})) == {"a": [1, 2, {3}], "b": (1.5, "x")}


def test_integrity_detects_every_tamper_type(tmp_path):
    (tmp_path / "w.bin").write_bytes(b"weights" * 100)
    (tmp_path / "cfg.json").write_text("{}")
    key = b"k"
    sign_dir(str(tmp_path), key)
    assert verify_dir(str(tmp_path), key) == []
    b = bytearray((tmp_path / "w.bin").read_bytes())
    b[10] ^= 1
    (tmp_path / "w.bin").write_bytes(bytes(b))
    assert verify_dir(str(tmp_path), key) == ["modified file: w.bin"]
    (tmp_path / "evil.pkl").write_bytes(b"x")
    os.remove(tmp_path / "cfg.json")
    p = verify_dir(str(tmp_path), key)
    assert "missing file: cfg.json" in p and "unexpected new file: evil.pkl" in p
    assert any("signature invalid" in x for x in verify_dir(str(tmp_path), b"wrong"))


@pytest.fixture(scope="module")
def data():
    X, y = make_dataset(4000, 64, 5, noise=2.5, seed=0)
    return X[:3000], y[:3000], X[3000:], y[3000:]


def test_attacks_reduce_accuracy_monotonically_and_pgd_beats_fgsm(data):
    Xtr, ytr, Xte, yte = data
    m = SoftmaxModel(64, 5).fit(Xtr, ytr)
    assert m.accuracy(Xte, yte) > 0.9
    pg = robust_accuracy(m, Xte, yte, [0, 0.25, 0.5, 1.0], "pgd")
    fg = robust_accuracy(m, Xte, yte, [0, 0.25, 0.5, 1.0], "fgsm")
    assert pg[0] > pg[0.25] > pg[0.5] > pg[1.0]
    assert pg[0.5] <= fg[0.5] + 0.02


def test_perturbation_stays_inside_linf_ball(data):
    Xtr, ytr, Xte, yte = data
    m = SoftmaxModel(64, 5).fit(Xtr, ytr)
    assert np.abs(pgd(m, Xte, yte, 0.3) - Xte).max() <= 0.3 + 1e-9


def test_adversarial_training_improves_robust_accuracy(data):
    Xtr, ytr, Xte, yte = data
    std = SoftmaxModel(64, 5).fit(Xtr, ytr)
    adv = adversarial_train(Xtr, ytr, 5, eps=0.5)
    assert robust_accuracy(adv, Xte, yte, [0.5])[0.5] > robust_accuracy(std, Xte, yte, [0.5])[0.5]


def test_backdoor_weight_check_has_signal_but_is_imperfect():
    tp = fp = 0
    for seed in range(10):
        X, y = make_dataset(4000, 64, 5, noise=2.5, seed=seed)
        Xp, yp = poison(X[:3000], y[:3000], 0, 0.1, [60, 61, 62, 63], 5.0, seed=seed)
        wb = weight_outliers(SoftmaxModel(64, 5).fit(Xp, yp))
        wc = weight_outliers(SoftmaxModel(64, 5).fit(X[:3000], y[:3000]))
        tp += bool(set(wb.get(0, [])) & {60, 61, 62, 63})
        fp += bool(wc)
    assert tp >= 5 and fp <= 3 and tp > fp


def test_cli_scan_exit_codes(tmp_path):
    bad, good = tmp_path / "bad.pkl", tmp_path / "good.pkl"
    bad.write_bytes(pickle.dumps(Evil(os.system, "x")))
    good.write_bytes(pickle.dumps({"a": 1}))
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run = lambda f: subprocess.run([sys.executable, "-m", "modelguard", "scan", str(f)], cwd=root,
                                   capture_output=True, text=True).returncode
    assert run(bad) == 1 and run(good) == 0
