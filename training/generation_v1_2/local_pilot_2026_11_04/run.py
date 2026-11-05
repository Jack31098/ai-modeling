"""Sequential, task-owned llama.cpp Vulkan teacher comparison, with raw receipts."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
import urllib.error
import urllib.request

from .validate import validate

BASE = Path(__file__).parent.resolve()
ROOT = BASE.parents[2]
RESULTS = BASE / "results"
PARAMS = {"temperature": 0, "top_k": 1, "top_p": 1, "seed": 7391, "max_tokens": 2048,
          "stream": False, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
COMMON_FLAGS = ["--host", "127.0.0.1", "--n-gpu-layers", "999", "--ctx-size", "8192",
                "--parallel", "1", "--threads", "8", "--batch-size", "512", "--ubatch-size", "512",
                "--flash-attn", "auto", "--reasoning", "off", "--reasoning-budget", "0",
                "--chat-template-kwargs", '{"enable_thinking":false}', "--no-context-shift", "--cache-ram", "0"]


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request(port, route, payload=None, timeout=600):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=data,
                                 headers={"Content-Type": "application/json"})
    # Explicitly avoid system proxies even for localhost.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as res:
        return json.loads(res.read())


def frozen_protocol():
    spec = {"protocol_version": "1.0", "created_before_evaluation": True,
            "params": PARAMS, "common_server_flags": COMMON_FLAGS,
            "prompt_sha256": sha(BASE / "prompt.txt"), "schema_sha256": sha(BASE / "schema.json"),
            "inputs_sha256": sha(BASE / "inputs/all100.jsonl"),
            "initial_cohort_size": 28, "expansion_total_size": 100,
            "maximum_attempts_per_doc": 2,
            "retry_policy": "Only retry a hard-invalid/truncated output; same original request plus exact error feedback and failed assistant output. Preserve both attempts.",
            "selection_policy": "first hard-valid output, otherwise final failure; never semantic best-of-N",
            "reasoning": "disabled uniformly; observe effective template and returned reasoning separately",
            "speed_scope": "single-slot Windows Vulkan Q8_0 local pilot; not MI300 throughput"}
    target = BASE / "protocol.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != spec:
            raise RuntimeError("Frozen protocol differs; make a new experiment rather than overwrite")
    else:
        write(target, spec)
    return spec


def run_model(model, docs, port):
    target = RESULTS / model["id"]
    target.mkdir(parents=True, exist_ok=True)
    lock = json.loads((BASE / "assets.lock.json").read_text(encoding="utf-8"))
    backend = ROOT / lock["backend"]["extract_path"]
    exe = backend / "llama-server.exe"
    model_path = ROOT / model["local_path"]
    receipt = ROOT / "models/local-teacher-pilot/verifications" / (model["id"] + ".json")
    if not receipt.exists() or model_path.stat().st_size != model["size_bytes"]:
        raise RuntimeError("Download/verification must complete before inference: " + model["id"])
    # Full SHA was already computed by the downloader. Keep the receipt with the committed result.
    write(target / "asset-verification.json", json.loads(receipt.read_text(encoding="utf-8")))
    # Verbosity is diagnostic only; the frozen generation/server-compute settings are unchanged.
    command = [str(exe), "--model", str(model_path), "--port", str(port), "--log-verbosity", "4"] + COMMON_FLAGS
    pending = []
    for doc in docs:
        result_path = target / "documents" / (doc["doc_id"].replace(":", "__") + ".json")
        if not result_path.exists():
            pending.append((doc, result_path))
    if not pending:
        print("COMPLETE existing", model["id"], flush=True)
        return
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
    env = os.environ.copy()
    trusted_runtime = Path(r"C:\Users\Minzh\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin")
    env["PATH"] = os.pathsep.join([str(trusted_runtime), sys.base_prefix, str(backend), env.get("PATH", "")])
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = "server-" + stamp
    meta = {"started_utc": now(), "command": command, "backend_executable_sha256": sha(exe),
            "model_id": model["id"], "model_asset": model, "protocol_sha256": sha(BASE / "protocol.json"),
            "runtime_path_additions": [str(trusted_runtime), sys.base_prefix, str(backend)],
            "owned_process": True, "pending_documents": len(pending)}
    log_path = target / (run_id + ".log")
    server = None
    start = time.monotonic()
    try:
        with log_path.open("wb") as log:
            server = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
                                      creationflags=subprocess.CREATE_NO_WINDOW)
            meta["pid"] = server.pid
            write(target / (run_id + ".json"), meta)
            print("SERVER_START", model["id"], server.pid, flush=True)
            deadline = start + 600
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Owned server exited {server.returncode}; inspect {log_path}")
                try:
                    health = request(port, "/health", timeout=2)
                    if health.get("status") == "ok":
                        break
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server load exceeded 600 seconds")
                time.sleep(1)
            meta.update(ready_utc=now(), load_wall_seconds=time.monotonic() - start,
                        props=request(port, "/props"), health=health)
            write(target / (run_id + ".json"), meta)
            print("SERVER_READY", model["id"], round(meta["load_wall_seconds"], 2), flush=True)
            schema = json.loads((BASE / "schema.json").read_text(encoding="utf-8"))
            prompt = (BASE / "prompt.txt").read_text(encoding="utf-8")
            for doc, result_path in pending:
                messages = [{"role": "system", "content": prompt},
                            {"role": "user", "content": json.dumps({"text": doc["text"]}, ensure_ascii=False)}]
                attempts = []
                for number in (1, 2):
                    payload = {**PARAMS, "messages": messages,
                               "response_format": {"type": "json_schema", "json_schema": {"name": "phrases", "strict": True, "schema": schema}}}
                    attempt = {"attempt": number, "started_utc": now(), "request": payload}
                    tick = time.monotonic()
                    try:
                        response = request(port, "/v1/chat/completions", payload)
                        attempt["response"] = response
                        choice = response["choices"][0]
                        message = choice["message"]
                        content = message.get("content") or ""
                        checked = validate(doc, content)
                        if choice.get("finish_reason") == "length":
                            checked.update(valid=False, errors=checked["errors"] + ["output_truncated"])
                        attempt["validation"] = checked
                        attempt["returned_reasoning_characters"] = len(message.get("reasoning_content") or "")
                    except Exception as exc:
                        attempt["transport_error"] = repr(exc)
                        attempt["validation"] = {"valid": False, "errors": ["transport_or_protocol_error"]}
                        content = ""
                    attempt.update(completed_utc=now(), wall_seconds=time.monotonic() - tick)
                    attempts.append(attempt)
                    write(target / "attempts" / (doc["doc_id"].replace(":", "__") + f"-{number}.json"), attempt)
                    print("RESULT", model["id"], doc["doc_id"], number,
                          attempt["validation"]["valid"], round(attempt["wall_seconds"], 2),
                          attempt["validation"]["errors"], flush=True)
                    if attempt["validation"]["valid"]:
                        break
                    messages = messages + [{"role": "assistant", "content": content},
                        {"role": "user", "content": "The annotation failed these hard checks: " +
                         json.dumps(attempt["validation"]["errors"], ensure_ascii=False) +
                         ". Produce the full corrected JSON for the exact original text. Follow the same budgets; no normalization or omitted spaces."}]
                write(result_path, {"doc_id": doc["doc_id"], "model_id": model["id"], "source_sha256": doc["content_sha256"],
                                    "cohort": doc["cohort"], "server_run_id": run_id,
                                    "attempt_count": len(attempts), "selected_attempt": len(attempts),
                                    "valid": attempts[-1]["validation"]["valid"],
                                    "attempt_files": ["../attempts/" + doc["doc_id"].replace(":", "__") + f"-{a['attempt']}.json" for a in attempts],
                                    "validation": attempts[-1]["validation"], "training_eligible": False})
    finally:
        if server is not None:
            # Stop only the exact Popen created above, including on failures or Ctrl-C.
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10)
            meta.update(stopped_utc=now(), exit_code=server.returncode, total_wall_seconds=time.monotonic() - start)
            write(target / (run_id + ".json"), meta)
            print("SERVER_STOP", model["id"], server.returncode, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--count", type=int, choices=[28, 100], default=28)
    parser.add_argument("--port", type=int, default=18471)
    args = parser.parse_args()
    frozen_protocol()
    lock = json.loads((BASE / "assets.lock.json").read_text(encoding="utf-8"))
    models = [m for m in lock["models"] if m["id"] == args.model]
    if len(models) != 1:
        raise ValueError("Select one locked model id")
    docs = [json.loads(line) for line in (BASE / "inputs/all100.jsonl").read_text(encoding="utf-8").splitlines()][:args.count]
    run_model(models[0], docs, args.port)


if __name__ == "__main__":
    main()
