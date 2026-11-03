# SCFP major revision, private Kaggle GPU job "gpub". Resumes from an attached previous output
# (files under results/revision/gpuB are restored; finished steps are skipped and logged calls reused).
import glob, os, shutil, subprocess, zipfile
WORK = "/kaggle/working"
PKG = os.path.join(WORK, "pkg")
z = glob.glob("/kaggle/input/**/scfp_revision_package.zip", recursive=True)
if z:
    zipfile.ZipFile(z[0]).extractall(WORK)
else:
    src = glob.glob("/kaggle/input/**/pkg/requirements_gpu.txt", recursive=True)
    if not src:
        raise SystemExit("attach the scfp-revision-package dataset")
    shutil.copytree(os.path.dirname(src[0]), PKG, dirs_exist_ok=True)
OUT = os.path.join(PKG, "results/revision/gpuB")
os.makedirs(OUT, exist_ok=True)
prev = [d for d in glob.glob("/kaggle/input/**/job_out", recursive=True) if os.path.isdir(d)]
if prev:
    shutil.copytree(prev[0], OUT, dirs_exist_ok=True)
    print("resumed from", prev[0], flush=True)
subprocess.run("nvidia-smi", shell=True)
subprocess.run(f"pip -q install -r {PKG}/requirements_gpu.txt", shell=True, check=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# stop before 6 h (author's per-job limit); finished parts are kept and the job can be resumed
r = subprocess.run(f"cd {PKG} && timeout 21000 python scripts/revision/gpu_external_judges.py --out results/revision/gpuB 2>&1 | grep --line-buffered -vE 'it/s\\]|Loading weights'", shell=True)
print("exit:", r.returncode, flush=True)
shutil.copytree(OUT, os.path.join(WORK, "job_out"), dirs_exist_ok=True)
print("FILES:", sorted(os.listdir(os.path.join(WORK, "job_out")))[:50], flush=True)
