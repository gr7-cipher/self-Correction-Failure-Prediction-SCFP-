# SCFP major revision M7: retrain the Table 8 encoders and Table 9 ablations with seed 4 (private Kaggle job).
import glob, os, shutil, subprocess, zipfile
SEED = 4
WORK = "/kaggle/working"
PKG = os.path.join(WORK, "pkg")
z = glob.glob("/kaggle/input/**/scfp_revision_train.zip", recursive=True)
if z:
    zipfile.ZipFile(z[0]).extractall(WORK)
else:
    src = glob.glob("/kaggle/input/**/pkg/requirements_gpu.txt", recursive=True)
    if not src:
        raise SystemExit("attach the scfp-revision-train dataset")
    shutil.copytree(os.path.dirname(src[0]), PKG, dirs_exist_ok=True)
RD = os.path.join(PKG, "SCFP/benchmark/scfp_runs/laptop_v1")
for f in glob.glob("/kaggle/input/**/seed_results/results/*__seed*.json", recursive=True):  # resume
    os.makedirs(os.path.join(RD, "results"), exist_ok=True); shutil.copy(f, os.path.join(RD, "results"))
for f in glob.glob("/kaggle/input/**/seed_results/predictions/*__seed*.jsonl", recursive=True):
    os.makedirs(os.path.join(RD, "predictions"), exist_ok=True); shutil.copy(f, os.path.join(RD, "predictions"))
subprocess.run("nvidia-smi", shell=True)
subprocess.run(f"pip -q install -r {PKG}/requirements_gpu.txt", shell=True, check=True)
r = subprocess.run(f"cd {PKG} && timeout 21000 bash scripts/revision/m7_train_seeds.sh {SEED} 2>&1 "
                   "| grep --line-buffered -vE 'it/s\\]|Loading weights'", shell=True)
print("exit:", r.returncode, flush=True)
out = os.path.join(WORK, "seed_results")
for sub in ("results", "predictions"):
    os.makedirs(os.path.join(out, sub), exist_ok=True)
    for f in glob.glob(os.path.join(RD, sub, f"*__seed{SEED}*")):
        if not f.endswith(".pt"):
            shutil.copy(f, os.path.join(out, sub))
n = len(glob.glob(os.path.join(out, "results", "*.json")))
print(f"RESULTS: {n}/10", "COMPLETE" if n == 10 else "PARTIAL: re-run with this output attached", flush=True)
