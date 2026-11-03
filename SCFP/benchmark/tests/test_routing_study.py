from scfp_bench.experiments import routing as R


def test_safe_program_runs():
    out = R.run_program("```python\nimport math\nprint(int(12 * 3 + math.sqrt(16)))\n```")
    assert out["stdout"].strip() == "40" and out["error"] is None


def test_unsafe_programs_are_rejected():
    for code in ["import os\nos.system('ls')", "open('/etc/passwd').read()",
                 "__import__('os')", "x = ().__class__", "import subprocess"]:
        assert R.run_program(f"```python\n{code}\n```")["error"].startswith("rejected")


def test_runaway_program_times_out():
    assert R.run_program("```python\nwhile True: pass\n```", timeout=1)["error"] == "timeout"
