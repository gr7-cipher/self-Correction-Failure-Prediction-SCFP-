# SCFP natural-set run on a Kaggle GPU (pushed with the Kaggle CLI).
# Copies the package from the attached dataset, restores a previous results zip if one is
# attached (resume), runs run_signals.sh with a time guard, and always zips what exists.
import glob, os, shutil, subprocess, zipfile

WORK = "/kaggle/working"
PKG = os.path.join(WORK, "SCFP_GPU_package")
src = glob.glob("/kaggle/input/**/run_signals.sh", recursive=True)
zips = glob.glob("/kaggle/input/**/SCFP_GPU_package.zip", recursive=True)
if src:
    shutil.copytree(os.path.dirname(src[0]), PKG, dirs_exist_ok=True)
elif zips:
    zipfile.ZipFile(zips[0]).extractall(WORK)
else:
    raise SystemExit("attach the scfp-gpu-package dataset")
prev = glob.glob("/kaggle/input/**/scfp_signals_results.zip", recursive=True)
prev_dir = glob.glob("/kaggle/input/**/scfp_runs/signals_v2/calls.jsonl", recursive=True)
if prev:
    zipfile.ZipFile(prev[0]).extractall(PKG)
    print("resumed from", prev[0], flush=True)
elif prev_dir:  # Kaggle may unpack an uploaded zip
    shutil.copytree(os.path.dirname(prev_dir[0]), os.path.join(PKG, "benchmark/scfp_runs/signals_v2"),
                    dirs_exist_ok=True)
    print("resumed from", os.path.dirname(prev_dir[0]), flush=True)
else:
    print("no previous run attached: starting fresh", flush=True)

# results are recomputed every session; drop restored ones so COMPLETE below means this session made them
shutil.rmtree(os.path.join(PKG, "benchmark/scfp_runs/signals_v2/results"), ignore_errors=True)
subprocess.run("nvidia-smi", shell=True)
subprocess.run(f"pip -q install -r {PKG}/requirements_gpu.txt", shell=True, check=True)
# Kaggle stops a session at 12 h and may then keep no output, so stop at 11 h 15 min.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
r = subprocess.run(f"cd {PKG} && timeout 40500 bash run_signals.sh 2>&1 | grep --line-buffered -vE 'it/s\\]|Loading weights'",
                   shell=True)
print("run_signals exit:", r.returncode, flush=True)
os.chdir(PKG)
subprocess.run("rm -f scfp_signals_results.zip && zip -qr scfp_signals_results.zip benchmark/scfp_runs/signals_v2 -x '*.npz'",
               shell=True)
shutil.copy(os.path.join(PKG, "scfp_signals_results.zip"), WORK)
done = os.path.exists(os.path.join(PKG, "benchmark/scfp_runs/signals_v2/results/improve_summary.md"))
print("COMPLETE" if done else "PARTIAL: re-run with this output attached to resume", flush=True)
