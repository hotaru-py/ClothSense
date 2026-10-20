"""Download the registration-gated models for the DeClotH rework (run it yourself; it asks for your password).

    .venv\\Scripts\\python tools\\fetch_mpi.py            # SMPL + ECON data
    .venv\\Scripts\\python tools\\fetch_mpi.py smpl       # only SMPL
    .venv\\Scripts\\python tools\\fetch_mpi.py econ       # only ECON (normal estimator weights)

Needs accounts (same e-mail + password works) on https://smpl.is.tue.mpg.de, https://smplify.is.tue.mpg.de and
https://icon.is.tue.mpg.de, each with its licence accepted. Files as in ECON's fetch_data.sh.
Writes models/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl and work/ECON/data/ (ckpt/normal.ckpt, smpl_data/ ...).
The password is only sent to download.is.tue.mpg.de and never stored.
"""
import getpass
import shutil
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://download.is.tue.mpg.de/download.php?domain={domain}&sfile={sfile}&resume=1"


def fetch(domain, sfile, dest, user, pw):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and zipfile.is_zipfile(dest):
        print(f"  have {dest.name}")
        return dest
    print(f"  downloading {domain}/{sfile} ...")
    with requests.post(URL.format(domain=domain, sfile=sfile), data={"username": user, "password": pw},
                       stream=True, timeout=60) as r:
        if r.status_code == 401:
            raise PermissionError(f"{domain}/{sfile}: 401 Unauthorized (wrong login, account not activated, or the "
                                  f"licence on https://{domain}.is.tue.mpg.de not accepted)")
        r.raise_for_status()
        if "html" in r.headers.get("content-type", ""):
            sys.exit(f"  {domain}: got a web page instead of the file -> wrong login, or the licence on "
                     f"https://{domain}.is.tue.mpg.de is not accepted for this account.")
        total, done = int(r.headers.get("content-length", 0)), 0
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
    tmp.replace(dest)
    if not zipfile.is_zipfile(dest):
        sys.exit(f"  {dest} is not a zip; delete it and retry")
    return dest


def smpl(user, pw):
    """Same files as ECON's fetch_data.sh: SMPL v1.0.0 (male, female; 10 shape PCs) + the neutral model from
    SMPLify (mpips_smplify_public_v2.zip). v1.1.0 gave 401 for the owner's account."""
    dl = ROOT / "work/downloads"
    if (dl / "SMPL_python_v.1.1.0.zip").exists():       # downloaded by hand: has all three genders
        return unpack_smpl([dl / "SMPL_python_v.1.1.0.zip"])
    unpack_smpl([fetch("smpl", "SMPL_python_v.1.0.0.zip", dl / "SMPL_python_v.1.0.0.zip", user, pw),
                 fetch("smplify", "mpips_smplify_public_v2.zip", dl / "mpips_smplify_public_v2.zip", user, pw)])


def unpack_smpl(zips):
    out = ROOT / "models/smpl"
    out.mkdir(parents=True, exist_ok=True)
    names = {"neutral": "SMPL_NEUTRAL.pkl", "_m_": "SMPL_MALE.pkl", "_f_": "SMPL_FEMALE.pkl"}
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if not n.endswith(".pkl") or "__MACOSX" in n or Path(n).name.startswith("._"):
                    continue                               # skip macOS resource forks (._*.pkl junk)
                for key, name in names.items():
                    if key in Path(n).name.lower() and "lbs_10_207" in n:
                        with zf.open(n) as src, open(out / name, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        print(f"  {n} -> models/smpl/{name}")
    missing = [n for n in names.values() if not (out / n).exists()]
    if missing:
        sys.exit(f"  missing after unzip: {missing}")
    sys.path.insert(0, str(ROOT / "tools"))
    from clean_smpl_pkl import clean                      # chumpy objects -> numpy (chumpy breaks on numpy 2)
    for n in names.values():
        print(f"  {n}: chumpy -> numpy for {clean(out / n) or 'nothing'}")


def econ(user, pw):
    try:                                                 # ECON's fetch_data.sh
        z = fetch("icon", "econ_data.zip", ROOT / "work/downloads/econ_data.zip", user, pw)
    except PermissionError as e:                         # ICON's fetch_data.sh (its normal net, same architecture)
        print(f"  {e}; trying icon_data.zip instead")
        z = fetch("icon", "icon_data.zip", ROOT / "work/downloads/icon_data.zip", user, pw)
    data = ROOT / "work/ECON/data"
    data.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(data)
    if (data / "smpl_data").exists():                    # same move as ECON's fetch_data.sh
        (data / "smpl_related").mkdir(exist_ok=True)
        dst = data / "smpl_related/smpl_data"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(data / "smpl_data"), str(dst))
    ck = data / "ckpt/normal.ckpt"
    print(f"  {'ok' if ck.exists() else 'MISSING'}: {ck.relative_to(ROOT)}")


if __name__ == "__main__":
    what = sys.argv[1:] or ["smpl", "econ"]
    user = input("MPI account e-mail: ").strip()
    pw = getpass.getpass("Password: ")
    for w in what:
        print(w)
        try:
            {"smpl": smpl, "econ": econ}[w](user, pw)
        except PermissionError as e:
            print(f"  FAILED: {e}")
    print("done")
