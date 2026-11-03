# SCFP major revision, GPU jobs on a Kaggle T4 (pushed with the Kaggle CLI; private).
# Unpacks the package from the attached private dataset, restores earlier call logs if a previous output
# is attached (resume: logged calls are never regenerated), runs the job with a time guard, and always
# zips what exists.
import glob, os, shutil, subprocess, zipfile

WORK = "/kaggle/working"
PKG = os.path.join(WORK, "pkg")
JOB = "m6"
zips = glob.glob("/kaggle/input/**/scfp_revision_package.zip", recursive=True)
if zips:
    zipfile.ZipFile(zips[0]).extractall(WORK)
else:
    src = glob.glob("/kaggle/input/**/pkg/requirements_gpu.txt", recursive=True)
    if not src:
        raise SystemExit("attach the scfp-revision-package dataset")
    shutil.copytree(os.path.dirname(src[0]), PKG, dirs_exist_ok=True)
OUT = os.path.join(PKG, "results/revision/m6/gen")
os.makedirs(OUT, exist_ok=True)
for f in glob.glob("/kaggle/input/**/m6_gen/calls_*.jsonl", recursive=True):
    shutil.copy(f, OUT)
    print("resumed calls from", f, flush=True)
subprocess.run("nvidia-smi", shell=True)
subprocess.run(f"pip -q install -r {PKG}/requirements_gpu.txt", shell=True, check=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Kaggle stops a session at 12 h and may then keep no output: stop generating at 11 h.
r = subprocess.run(f"cd {PKG} && python scripts/revision/m6_sc_generate.py --k 3 --batch-size 8 "
                   f"--out results/revision/m6/gen --deadline-s 39600 2>&1 | grep --line-buffered -vE 'it/s\\]|Loading weights'",
                   shell=True)
print("exit:", r.returncode, flush=True)
dst = os.path.join(WORK, "m6_gen")
shutil.copytree(OUT, dst, dirs_exist_ok=True)
done = sorted(os.path.basename(f) for f in glob.glob(os.path.join(dst, "done_*.json")))
print("DONE FILES:", done, flush=True)
print("COMPLETE" if len(done) == 4 else "PARTIAL: re-run with this output attached to resume", flush=True)
