import os, sys, subprocess, threading, io

def tee_stream(stream, fileobj, prefix=""):
    for line in iter(stream.readline, b""):
        try:
            decoded = line.decode("utf-8", "replace")
        except Exception:
            decoded = line.decode(errors="replace")
        msg = f"{prefix} {decoded}" if prefix else decoded
        print(msg, end="")
        fileobj.write(msg)
        fileobj.flush()
    stream.close()

def launch(script: str, cuda_id: str, log_path: str, name: str):
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = cuda_id

    p = subprocess.Popen(
        [sys.executable, "-u", script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=True,
    )
    f = open(log_path, "w", buffering=1)  # line-buffered file
    t = threading.Thread(target=tee_stream, args=(p.stdout, f, f"[{name}]"), daemon=True)
    t.start()
    return {"p": p, "f": f, "t": t, "name": name, "log": log_path}

def run_sequence(sequence_name: str, cuda_id: str, jobs, results_sink: dict):
    """
    Run a list of (script, log_path, display_name) sequentially on the same CUDA device.
    Store per-job exit codes in results_sink[sequence_name].
    """
    print(f"\n=== Starting sequence {sequence_name} on CUDA:{cuda_id} ===")
    seq_results = []
    for script, log_path, disp in jobs:
        print(f"[{sequence_name}] Launching {disp}: {script} (CUDA:{cuda_id}), log: {log_path}")
        j = launch(script, cuda_id, log_path, f"{sequence_name}:{disp}")
        rc = j["p"].wait()
        j["t"].join()
        j["f"].close()
        print(f"[{sequence_name}] Finished {disp} with exit code {rc}")
        seq_results.append({"name": j["name"], "rc": rc, "log": j["log"]})
    results_sink[sequence_name] = seq_results
    print(f"=== Sequence {sequence_name} complete ===\n")

def tail(path, n=60):
    with open(path, "rb") as f:
        f.seek(0, io.SEEK_END)
        size = f.tell()
        block, data = 2048, b""
        while size > 0 and data.count(b"\n") <= n:
            step = min(block, size)
            size -= step
            f.seek(size)
            data = f.read(step) + data
        return data.decode("utf-8", "replace").splitlines()[-n:]

def main():
    # Define per-GPU sequential job lists
    gpu0_jobs = [
        ("train_qwen3_14b.py", "gpu0_qwen3_14b.log", "qwen3-14b"),
        #("train_phi4.py", "gpu0_phi4.log", "phi4"),
        ("train_qwen3_8b.py", "gpu0_qwen3_8b.log", "qwen3-8b"),
        ("train_qwen3_4b.py", "gpu0_qwen3_4b.log", "qwen3-4b"),
        #("train_llama3_3b.py", "gpu0_llama3_3b.log", "llama3-3b"),
    ]
    gpu1_jobs = [
        ("train_qwen2.5_14b.py", "gpu0_qwen2_14b.log", "qwen2-14b"),
        #("train_qwen3_8b.py", "gpu0_qwen3_8b.log", "qwen3-8b"),
        #("train_phi4.py", "gpu0_phi4.log", "phi4"),
        #("train_qwen3_4b.py", "gpu0_qwen3_4b.log", "qwen3-4b"),
        ("train_llama3_8b.py", "gpu0_llama3_8b.log", "llama3-8b"),
        #("train_llama3_3b.py", "gpu0_llama3_3b.log", "llama3-3b"),
        ("train_ettin_400m.py", "gpu0_ettin_400m.log", "ettin-400m"),
    ]

    print("Launching training sequences:")
    print(" - GPU0: qwen3-14b")
    print(" - GPU1: qwen2.5-14b")

    results = {}
    t0 = threading.Thread(target=run_sequence, args=("GPU0", "0", gpu0_jobs, results), daemon=True)
    t1 = threading.Thread(target=run_sequence, args=("GPU1", "1", gpu1_jobs, results), daemon=True)
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    # Summarize exit codes
    print("\n=== Exit codes ===")
    for seq_name in ("GPU0", "GPU1"):
        for item in results.get(seq_name, []):
            print(f"{item['name']}: {item['rc']}")

    # Quick tails for convenience
    for seq_name in ("GPU0", "GPU1"):
        for item in results.get(seq_name, []):
            print(f"\n----- {item['name']} tail ({item['log']}) -----")
            try:
                print("\n".join(tail(item["log"], 80)))
            except FileNotFoundError:
                print("(log file not found)")

if __name__ == "__main__":
    main()