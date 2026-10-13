"""Download SKM-TEA (DICOM archives) or MRNet (.npy volumes + labels) from Redivis, run from src/.

    REDIVIS_API_TOKEN=... python -m ext.download --data skm      # -> input/skm_tea/dicom/MTR_*.tar.gz (155, 19 GB)
    REDIVIS_API_TOKEN=... python -m ext.download --data mrnet    # -> input/mrnet/{train,valid}/{axial,coronal,sagittal}/*.npy + 6 csv

Needs `pip install redivis` and a Redivis API token with access to the AIMI datasets (accepted data use
agreements). The token is read from the environment, never written anywhere. SKM-TEA labels
(train.json / val.json / test.json, scan_labels.csv) are a separate download from the AIMI page, into
input/skm_tea/.

Redivis notes: parallel downloads over HTTP/3 crash (BufferReadError) and listings time out, hence the
retries (and sequential SKM downloads); MRNet file names must come from `properties["file_name"]`
(`f.name` drops the plane folder, so planes overwrite each other); the label tables come back with their
first row as the header, which is restored.
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from params import INPUT_FOLDER

SKM_TABLE = "aimi.skm_tea:5r8z:v1_0.dicom:efwx"
MRNET = "aimi.mrnet_knee_mri_s:4a2c:v1_0"
MRNET_LABELS = ["train_acl:v9mj", "train_meniscus:3x4c", "train_abnormal:4v5w",
                "valid_acl:sqwh", "valid_meniscus:nqng", "valid_abnormal:851r"]
MRNET_IMAGES = [("valid", "valid:zcmk"), ("train", "train:5wjf")]


def retry(fn, what, n=8):
    for a in range(n):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            print(f"{what}: attempt {a + 1} failed: {type(e).__name__}", flush=True)
            time.sleep(5 * (a + 1))
    raise RuntimeError(f"{what} failed")


def skm():
    import redivis

    out = os.path.join(INPUT_FOLDER, "skm_tea", "dicom")
    os.makedirs(out, exist_ok=True)
    files = retry(lambda: redivis.table(SKM_TABLE).list_files(max_results=500), "listing")
    failed = []
    for i, f in enumerate(files):
        dst = os.path.join(out, f.name)
        if os.path.exists(dst) and os.path.getsize(dst) == f.size:
            continue
        try:
            retry(lambda: f.download(out + os.sep, overwrite=True, progress=False), f.name, n=5)
        except RuntimeError:
            failed.append(f.name)
        print(f"{i + 1}/{len(files)}", flush=True)
    print("FINISHED; failed:", failed)


def mrnet():
    import redivis

    out = os.path.join(INPUT_FOLDER, "mrnet")
    os.makedirs(out, exist_ok=True)
    for t in MRNET_LABELS:
        df = retry(lambda: redivis.table(f"{MRNET}.{t}").to_pandas_dataframe(), t)
        if list(df.columns) != ["case", "label"]:        # first row read as the header: put it back
            df = pd.concat([pd.DataFrame([df.columns.tolist()], columns=df.columns), df])
            df.columns = ["case", "label"]
        df["case"] = df.case.astype(str).str.zfill(4)
        df.to_csv(os.path.join(out, t.split(":")[0].replace("_", "-") + ".csv"), index=False)
    for split, ref in MRNET_IMAGES:
        files = retry(lambda: redivis.table(f"{MRNET}.{ref}").list_files(max_results=10000), f"list {split}")

        def get(f):
            dst = os.path.join(out, split, f.properties["file_name"])      # keeps axial/ coronal/ sagittal/
            if os.path.exists(dst) and os.path.getsize(dst) == f.size:
                return True
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                retry(lambda: f.download(dst, overwrite=True, progress=False), f.name, n=5)
                return True
            except RuntimeError:
                return False

        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(get, files))
        print(f"{split}: {sum(res)}/{len(files)} files", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, choices=["skm", "mrnet"])
    a = p.parse_args()
    assert os.environ.get("REDIVIS_API_TOKEN"), "set REDIVIS_API_TOKEN"
    skm() if a.data == "skm" else mrnet()


if __name__ == "__main__":
    main()
