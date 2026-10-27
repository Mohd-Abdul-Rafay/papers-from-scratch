"""
Fetch CIFAR-10, check it against the published md5s, and extract it.

Written after a download that was the right SIZE and the wrong CONTENTS. The
archive had been pulled with 8 parallel connections straight into a
network-mounted folder, which preallocates the full file length and then fills
it in -- so a half-finished download looks complete to `ls`, and `tar -xzf`
unpacks whatever happens to be there.

So: download to local disk, verify, and only then put a copy anywhere else.
Nothing is deleted; a bad download is reported and left alone.

    python prepare_data.py --data ./dataset
    python prepare_data.py --data ./dataset --archive-copy /some/backup/dir

Paths come from the command line or from RESNET_DATA. No path is written into
this file.

md5s are torchvision's own (torchvision/datasets/cifar.py).
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import tarfile

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE = "cifar-10-python.tar.gz"
BATCH_DIR = "cifar-10-batches-py"

TAR_MD5 = "c58f30108f718f92721af3b95e74349a"
BATCH_MD5 = {
    "batches.meta": "5ff9c542aee3614f3951f8cda6e48888",
    "data_batch_1": "c99cafc152244af753f735de768cd75f",
    "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
    "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
    "data_batch_4": "634d18415352ddfa80567beed471001a",
    "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
    "test_batch":   "40351d587109b95175f43aff81a1287e",
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def batches_ok(batch_dir):
    for name, want in BATCH_MD5.items():
        p = os.path.join(batch_dir, name)
        if not os.path.exists(p) or md5(p) != want:
            return False
    return True


def download(dest_dir, attempts=3, connections=8):
    """Local disk only. Returns the path to a verified archive, or raises."""
    os.makedirs(dest_dir, exist_ok=True)
    tar = os.path.join(dest_dir, ARCHIVE)

    for attempt in range(1, attempts + 1):
        for junk in (tar, tar + ".aria2"):
            if os.path.exists(junk):
                os.remove(junk)

        if shutil.which("aria2c"):
            cmd = ["aria2c", f"-x{connections}", f"-s{connections}",
                   "--allow-overwrite=true", "--auto-file-renaming=false",
                   "-d", dest_dir, "-o", ARCHIVE, URL]
        elif shutil.which("curl"):
            cmd = ["curl", "-fL", "-o", tar, URL]
        elif shutil.which("wget"):
            cmd = ["wget", "-O", tar, URL]
        else:
            raise RuntimeError("no aria2c, curl or wget on PATH")
        subprocess.run(cmd, check=False)

        got = md5(tar) if os.path.exists(tar) else "no file was written"
        print(f"  attempt {attempt}: md5 {got}")
        if got == TAR_MD5:
            return tar
        # A leftover .aria2 control file means the download never finished,
        # whatever the file size says.
        if os.path.exists(tar + ".aria2"):
            print("  (a .aria2 control file is still here: the download was "
                  "incomplete, not corrupt in transit)")

    raise RuntimeError(f"{attempts} attempts, never got md5 {TAR_MD5}. "
                       f"Nothing was deleted; the files are in {dest_dir}.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--data", default=os.environ.get("RESNET_DATA", "./dataset"),
                    help="where cifar-10-batches-py should end up")
    ap.add_argument("--work", default="/tmp",
                    help="local scratch for the download (must be real local disk)")
    ap.add_argument("--archive-copy", default=None,
                    help="also keep the verified archive here")
    args = ap.parse_args()

    batch_dir = os.path.join(args.data, BATCH_DIR)
    if batches_ok(batch_dir):
        print(f"already present and verified: {batch_dir}")
    else:
        existing = os.path.join(args.data, ARCHIVE)
        if os.path.exists(existing) and md5(existing) == TAR_MD5:
            tar = existing
            print(f"archive already here and verified: {tar}")
        else:
            tar = download(args.work)
            os.makedirs(args.data, exist_ok=True)
            if os.path.abspath(os.path.dirname(tar)) != os.path.abspath(args.data):
                shutil.copy(tar, os.path.join(args.data, ARCHIVE))
                print(f"verified archive copied to {args.data}")

        os.makedirs(args.data, exist_ok=True)
        with tarfile.open(tar) as f:
            try:
                f.extractall(args.data, filter="data")
            except TypeError:           # python < 3.12 has no filter=
                f.extractall(args.data)
        if not batches_ok(batch_dir):
            raise RuntimeError(
                f"the archive verified but the extracted files did not. "
                f"Nothing deleted; look in {batch_dir}.")
        print("extracted and verified")

    for name in sorted(BATCH_MD5):
        p = os.path.join(batch_dir, name)
        print(f"  {name:<14} {os.path.getsize(p)/1e6:7.2f} MB  md5 ok")

    if args.archive_copy:
        src = os.path.join(args.data, ARCHIVE)
        if os.path.exists(src):
            os.makedirs(args.archive_copy, exist_ok=True)
            dest = os.path.join(args.archive_copy, ARCHIVE)
            shutil.copy(src, dest)
            if md5(dest) != TAR_MD5:
                raise RuntimeError(f"the copy at {dest} does not verify")
            print(f"archive copy verified at {dest}")


if __name__ == "__main__":
    main()
