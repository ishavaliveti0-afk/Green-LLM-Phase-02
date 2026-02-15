import json
import time
from pathlib import Path
from typing import Any, Dict, List
import sys

import requests

# ============================

import json
import time
import sys
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import requests

# ============================================================
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("WATSONX_API_KEY")

# 878_mistralai.py
# Runs Watsonx *chat* for prompt12/prompt13 with:
# correct message structure
# prompt-signature folders (fresh results when prompts change)
# resume + skip per-run (within signature folder)
# run-only option (run a single run_idx)
# force-fresh option (ignore old dataset.json and overwrite)

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
MODEL_ID = "mistralai/mistral-medium-2505"
API_VERSION = "2023-05-29"


NUM_RUNS = 10
SLEEP_BETWEEN_CALLS = 0.1

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset-878problems.json"
MAPPING_PATH = BASE_DIR / "problem_id_algorithm_mapping.json"
RESULTS_DIR = BASE_DIR / "results_878_v3.1"

PROMPT_TYPES = ["prompt12", "prompt13"]

IAM_URL = "https://iam.cloud.ibm.com/identity/token"
CHAT_URL = f"{WATSONX_URL}/ml/v1/text/chat?version={API_VERSION}"

# Chat parameters
GEN_PARAMS = {
    "max_new_tokens": 512,
    "temperature": 0.0,          # greedy-like
    "top_p": 1.0,
    "repetition_penalty": 1.05,
    "return_usage": True,
}


# Helpers
# ============================================================

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def get_iam_token() -> str:
    if not API_KEY or API_KEY.startswith("YOUR_"):
        raise RuntimeError("Set API_KEY near the top of 878_mistralai.py")
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": API_KEY,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(IAM_URL, data=data, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]

def get_problem_id(problem: Dict[str, Any]) -> Any:
    for key in ("id", "problem_id", "problemIdx", "problem_idx"):
        if key in problem:
            return problem[key]
    raise KeyError(f"Problem ID not found: keys={list(problem.keys())}")

def get_description(problem: Dict[str, Any]) -> str:
    for key in ("markdown_description", "description", "prompt", "question"):
        if key in problem:
            return str(problem[key])
    return json.dumps(problem, indent=2, ensure_ascii=False)

def get_category(problem_id: Any, mapping: Any) -> str:
    # Case 1: direct mapping id -> category
    if isinstance(mapping, dict):
        if str(problem_id) in mapping and isinstance(mapping[str(problem_id)], str):
            return mapping[str(problem_id)]
        # Case 2: category -> list of ids
        for category, ids in mapping.items():
            if isinstance(ids, list) and (problem_id in ids or str(problem_id) in ids):
                return category
    return "unknown"


# ============================================================

def build_messages(system_prompt: str, user_prompt: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": f"{system_prompt}",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"{user_prompt}\n"
                        f"Now the actual task for you: \n"
                        f"# Task description:\n```python\n{payload['description']}\n```\n"
                        f"# Test case:\n```python\n{payload['small_test_cases']}\n```"
                    ),
                }
            ],
        },
    ]


# ============================================================

system_prompt_12 = (
    "You are an expert competitive programmer focused on energy-efficient code. "
    "Solve the problem with the most energy-efficient algorithm and data structures possible, "
    "minimizing runtime, memory usage, and CPU cycles."
)
user_prompt_12 = (
    "Please based on the task description write Solution to pass the provided test cases.\n"
    "You must follow the following rules:\n"
    "1. The code should be in ```python\\n[Code]\\n``` block.\n"
    "2. Do not add the provided test cases into your ```python\\n[Code]\\n``` block.\n"
    "3. You do not need to write the test cases, we will provide the test cases for you.\n"
    "4. Make sure that the provided test cases can pass your solution.\n"
    "5. Prefer the energy-efficient solution from the provided examples of inefficient and efficient solutions.\n"
    "6. Always choose the most efficient algorithm in terms of both time and memory.\n"
)

system_prompt_13 = system_prompt_12
user_prompt_13 = (
    "Please based on the task description write Solution to pass the provided test cases.\n"
    "You must follow the following rules:\n"
    "1. The code should be in ```python\\n[Code]\\n``` block.\n"
    "2. You do not need to write the test cases, we will provide the test cases for you.\n"
    "3. Make sure that the provided test cases can pass your solution.\n"
    "4. Prefer the energy-efficient solution from the provided examples of inefficient and efficient solutions.\n"
    "5. Always choose the most efficient algorithm in terms of both time and memory.\n"
)

PROMPT_TEXTS: Dict[str, Tuple[str, str]] = {
    "prompt12": (system_prompt_12, user_prompt_12),
    "prompt13": (system_prompt_13, user_prompt_13),
}


# ============================================================

def extract_chat_text(resp_json: dict) -> str:
    # Common: {"choices":[{"message":{"role":"assistant","content":"..."}}], ...}
    choices = resp_json.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content

    # Sometimes: {"results":[{"generated_text":"..."}], ...}
    results = resp_json.get("results")
    if isinstance(results, list) and results:
        t = results[0].get("generated_text") or results[0].get("text")
        if isinstance(t, str):
            return t

    return ""

def generate_completion_chat(messages: List[Dict[str, Any]], token: str) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    body = {
        "model_id": MODEL_ID,
        "project_id": PROJECT_ID,
        "messages": messages,
        "parameters": GEN_PARAMS,
    }

    start = time.perf_counter()
    resp = requests.post(CHAT_URL, json=body, headers=headers, timeout=120)
    runtime_sec = time.perf_counter() - start

    if not resp.ok:
        return {
            "ok": False,
            "status_code": resp.status_code,
            "runtime_sec": runtime_sec,
            "text": "",
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "raw": resp.text,
        }

    data = resp.json()
    text = extract_chat_text(data)

    usage = data.get("usage") or {}
    in_t = int(usage.get("input_tokens") or 0)
    out_t = int(usage.get("output_tokens") or 0)
    tot_t = int(usage.get("total_tokens") or (in_t + out_t))

    return {
        "ok": True,
        "status_code": resp.status_code,
        "runtime_sec": runtime_sec,
        "text": text,
        "usage": {"input_tokens": in_t, "output_tokens": out_t, "total_tokens": tot_t},
        "raw": data,
    }


# ============================================================

def prompt_signature(system_prompt: str, user_prompt: str) -> str:
    h = hashlib.sha1((system_prompt + "\n" + user_prompt).encode("utf-8")).hexdigest()
    return h[:10]

def sanitize_model_name(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")

def run_prompt_type(
    prompt_type: str,
    problems: List[Dict[str, Any]],
    mapping: Any,
    *,
    num_runs: int,
    run_only: Optional[int],
    force_fresh: bool,
) -> None:
    if prompt_type not in PROMPT_TEXTS:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")

    sys_p, usr_p = PROMPT_TEXTS[prompt_type]
    sig = prompt_signature(sys_p, usr_p)

    model_clean = sanitize_model_name(MODEL_ID)

    # Signature-specific root => changing prompts creates a new folder automatically
    root = RESULTS_DIR / f"{prompt_type}_{sig}"
    root.mkdir(parents=True, exist_ok=True)

    print(f"[{prompt_type}] signature={sig} root={root.resolve()}")

    for run_idx in range(num_runs):
        # ✅ This is the ONLY place run_only is checked
        if run_only is not None and run_idx != run_only:
            continue

        run_dir = root / f"{model_clean}_{run_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "dataset.json"

        # Resume within THIS run folder unless force_fresh
        run_results: List[Dict[str, Any]] = []
        completed_ids: set[str] = set()

        if (not force_fresh) and out_path.exists():
            try:
                existing = load_json(out_path)
                if isinstance(existing, list):
                    run_results = existing
                    for rec in existing:
                        pid = rec.get("problem_id")
                        if pid is not None:
                            completed_ids.add(str(pid))
                    print(f"[{prompt_type}] Run {run_idx}: resuming with {len(completed_ids)} completed.")
                else:
                    print(f"[{prompt_type}] Run {run_idx}: dataset.json not a list, starting fresh.")
            except Exception as e:
                print(f"[{prompt_type}] Run {run_idx}: failed to read dataset.json ({e}), starting fresh.")
        else:
            if force_fresh and out_path.exists():
                print(f"[{prompt_type}] Run {run_idx}: FORCE_FRESH enabled -> ignoring existing dataset.json")
            else:
                print(f"[{prompt_type}] Run {run_idx}: starting fresh (no resume).")

        token = get_iam_token()

        for i, problem in enumerate(problems):
            problem_id = get_problem_id(problem)
            pid_str = str(problem_id)

            if pid_str in completed_ids:
                continue

            payload = {
                "description": get_description(problem),
                "small_test_cases": problem.get("small_test_cases", ""),
            }

            messages = build_messages(sys_p, usr_p, payload)
            gen_info = generate_completion_chat(messages, token)

            status = "ok" if gen_info["ok"] else "error"
            completion_text = gen_info["text"] if gen_info["ok"] else ""

            record = {
                "problem_id": problem_id,
                "algorithm_category": get_category(problem_id, mapping),
                "prompt_type": prompt_type,
                "prompt_signature": sig,
                "model_id": MODEL_ID,
                "run_idx": run_idx,
                "prompt_messages": messages,
                "completion": completion_text,
                "input_tokens": gen_info["usage"]["input_tokens"],
                "output_tokens": gen_info["usage"]["output_tokens"],
                "total_tokens": gen_info["usage"]["total_tokens"],
                "runtime_sec": gen_info["runtime_sec"],
                "status": status,
                "error_raw": (gen_info["raw"] if status == "error" else None),
            }

            run_results.append(record)
            completed_ids.add(pid_str)

            try:
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(run_results, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[{prompt_type}] Run {run_idx} i={i} id={problem_id} WARNING: failed to write ({e})")

            print(
                f"[{prompt_type}] Run {run_idx} i={i} id={problem_id} "
                f"status={status} in={record['input_tokens']} out={record['output_tokens']}"
            )

            time.sleep(SLEEP_BETWEEN_CALLS)

        print(f"[{prompt_type}] Run {run_idx}: finished. Total records: {len(run_results)}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt_type",
        nargs="?",
        default=None,
        help=f"Optional: prompt12 or prompt13. If omitted, runs {PROMPT_TYPES}.",
    )
    parser.add_argument(
        "--run",
        dest="run_only",
        type=int,
        default=None,
        help="Run only a specific run index (0-based). Example: --run 1",
    )
    parser.add_argument(
        "--num-runs",
        dest="num_runs",
        type=int,
        default=NUM_RUNS,
        help="How many runs to execute (default from NUM_RUNS).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing dataset.json (force fresh run output overwrite).",
    )
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(f"Mapping file not found at {MAPPING_PATH}")

    problems = load_json(DATASET_PATH)
    mapping = load_json(MAPPING_PATH)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.prompt_type is None:
        selected = PROMPT_TYPES
    else:
        if args.prompt_type not in PROMPT_TEXTS:
            raise SystemExit(f"Unknown prompt type: {args.prompt_type}. Choose from {list(PROMPT_TEXTS.keys())}")
        selected = [args.prompt_type]

    print(f"Loaded {len(problems)} problems.")
    print(f"Results dir: {RESULTS_DIR.resolve()}")
    print(f"Selected prompt types: {selected}")
    print(f"RUN_ONLY: {args.run_only} (None means all runs 0..{args.num_runs-1})")
    print(f"FORCE_FRESH: {args.fresh}")

    for pt in selected:
        print(f"\n=== Running prompt type: {pt} ===")
        run_prompt_type(
            pt,
            problems,
            mapping,
            num_runs=args.num_runs,
            run_only=args.run_only,
            force_fresh=args.fresh,
        )

    print("\nAll runs complete.")

if __name__ == "__main__":
    main()
