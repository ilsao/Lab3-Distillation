#!/usr/bin/env bash
# Run the Section 3.3 protocol without changing main.py or guidance/sd.py.
# Usage: ./ablation.sh [--device 0] [--dry-run] [--overwrite] [--eval-only]
# Select the interpreter with PYTHON=/path/to/python ./ablation.sh.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
runner=$(mktemp "${TMPDIR:-/tmp}/lab3-ablation.XXXXXX.py")
trap 'rm -f -- "$runner"' EXIT
cat > "$runner" <<'PY'
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

# The shell starts this helper from the repository root.
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
PROMPTS = (
    "A burger on the table",
    "a cat sitting on a table",
    "A villa close to the pool",
)
# One run per prompt and setting; seed is fixed, not another experiment axis.
SEEDS = (2024,)
SWEEPS = {
    "sds_guidance_scale": (7.5, 15, 25, 50, 100),
    "sdi_inversion_n_steps": (1, 3, 5, 10, 25),
    "sdi_update_interval": (25, 75, 200, 500, 1000),
}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_signature():
    paths = (ROOT / "main.py", ROOT / "guidance/sd.py", ROOT / "utils.py",
             ROOT / "eval.py", Path(__file__))
    return hashlib.sha256("".join(digest(p) for p in paths).encode()).hexdigest()


def implementation_paths():
    return (ROOT / "main.py", ROOT / "guidance/sd.py", ROOT / "utils.py", ROOT / "eval.py")


def implementation_signature():
    return hashlib.sha256("".join(digest(p) for p in implementation_paths()).encode()).hexdigest()


def canonical(group, value):
    # Both sweeps share N=10, update_interval=25. Generate it only once.
    if group == "sdi_update_interval" and value == 25:
        return "sdi_inversion_n_steps", 10
    return group, value


def saved_baseline(group, value):
    """Reuse complete baseline prompt sets; the old runs have no recorded seeds."""
    if (group, value) not in (("sds_guidance_scale", 25), ("sdi_inversion_n_steps", 10)):
        return None
    method = "sds" if group == "sds_guidance_scale" else "sdi"
    folder = ROOT / "outputs" / method
    try:
        config = json.loads((folder / "run_config.json").read_text())
        evaluation = json.loads((folder / "eval.json").read_text())
        required = dict(loss_type=method, precision="fp32", negative_prompt="low quality",
                        steps=500 if method == "sds" else 1000,
                        lr=0.01 if method == "sds" else 0.005,
                        guidance_scale=25 if method == "sds" else 7.5)
        if method == "sdi":
            required.update(inversion_n_steps=10, inversion_guidance_scale=-7.5,
                            sdi_update_interval=25)
        if any(config.get(key) != value for key, value in required.items()):
            return None
        scores = {Path(k).name: v for k, v in evaluation.items() if k.endswith(".png")}
        for prompt in PROMPTS:
            name = prompt.replace(" ", "_") + ".png"
            if not (folder / name).is_file() or not math.isfinite(scores[name]):
                return None
        return folder
    except (OSError, ValueError, KeyError, TypeError):
        return None


def make_jobs(output, device):
    jobs = {}
    for group, values in SWEEPS.items():
        for value in values:
            group_key, value_key = canonical(group, value)
            reference = saved_baseline(group_key, value_key)
            for seed in ((None,) if reference else SEEDS):
                for prompt in PROMPTS:
                    key = (group_key, value_key, seed, prompt)
                    if key in jobs:
                        continue
                    sds = group_key == "sds_guidance_scale"
                    jobs[key] = {
                        "group": group_key, "value": value_key, "seed": seed,
                        "prompt": prompt, "device": device,
                        "loss_type": "sds" if sds else "sdi",
                        "steps": 500 if sds else 1000,
                        "lr": 0.01 if sds else 0.005,
                        "guidance_scale": value_key if sds else 7.5,
                        "inversion_n_steps": value_key if group_key == "sdi_inversion_n_steps" else 10,
                        "inversion_guidance_scale": -7.5, "inversion_eta": 0.3,
                        "update_interval": value_key if group_key == "sdi_update_interval" else 25,
                        "negative_prompt": "low quality", "precision": "fp32",
                        "folder": str(output / group_key / str(value_key) /
                                      ("existing_unseeded" if reference else f"seed_{seed}")),
                    }
                    if reference:
                        jobs[key]["reuse_source"] = str(reference)
    return list(jobs.values())


def artifacts(job):
    stem = Path(job["folder"]) / job["prompt"].replace(" ", "_")
    return stem.with_suffix(".png"), stem.with_suffix(".json"), stem.with_suffix(".log")


def complete(job):
    image, metadata, _ = artifacts(job)
    try:
        data = json.loads(metadata.read_text())
        compatible = data["fingerprint"] == job["fingerprint"]
        if not compatible and "reuse_source" not in job:
            # Script-only changes (such as removing seed repetitions) must not
            # invalidate completed images with identical training settings.
            ignored = {"fingerprint", "implementation_signature"}
            stored = {k: v for k, v in data.get("config", {}).items() if k not in ignored}
            current = {k: v for k, v in job.items() if k not in ignored}
            recorded_signature = data.get("config", {}).get("implementation_signature")
            if recorded_signature is not None:
                same_code = recorded_signature == implementation_signature()
            else:
                # Older metadata predates explicit model-code hashes. Reuse it
                # only if the implementation files have not been edited since.
                same_code = all(p.stat().st_mtime <= metadata.stat().st_mtime
                                for p in implementation_paths())
            compatible = stored == current and same_code
        valid = compatible and data["image_sha256"] == digest(image)
        if "reuse_source" in job:
            original = Path(job["reuse_source"]) / image.name
            return valid and data.get("reused_existing", False) and digest(original) == digest(image)
        return valid and math.isfinite(data["optimization_seconds"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def import_baseline(job):
    image, metadata, _ = artifacts(job)
    folder = Path(job["reuse_source"])
    source = folder / image.name
    evaluation = json.loads((folder / "eval.json").read_text())
    score = next(v for k, v in evaluation.items() if Path(k).name == image.name)
    image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, image)
    write_json(metadata, {
        "fingerprint": job["fingerprint"], "config": job, "reused_existing": True,
        "source_image": str(source), "source_eval": str(folder / "eval.json"),
        "source_config": str(folder / "run_config.json"), "clip_score": score,
        "optimization_seconds": None, "initial_latent_sha256": None,
        "image_sha256": digest(image),
        "note": "Existing unseeded output; original implementation provenance and runtime unavailable.",
    })


def worker(job):
    # A fresh subprocess creates a fresh model/cache for every prompt and seed.
    import torch
    import main as training
    from utils import torch_to_pil

    torch.cuda.set_device(job["device"])
    training.seed_everything(job["seed"])
    args = SimpleNamespace(device=job["device"], precision=torch.float32,
                           loss_type=job["loss_type"], lora_rank=4)
    model = training.init_model(args)
    cond = model.get_text_embeds(job["prompt"])
    neg = model.get_text_embeds(job["negative_prompt"])
    embeddings = torch.cat([neg, cond])
    # Reset once more so latent initialization cannot depend on loader RNG use.
    training.seed_everything(job["seed"])
    latents = torch.nn.Parameter(torch.randn(1, 4, 64, 64,
                                           device=model.device, dtype=torch.float32))
    initial_hash = hashlib.sha256(latents.detach().cpu().numpy().tobytes()).hexdigest()
    optimizer = torch.optim.AdamW([latents], lr=job["lr"], weight_decay=0)
    schedule = training.get_cosine_schedule_with_warmup(
        optimizer, 100, int(job["steps"] * 1.5))

    # Time only optimization: no model loading, decoding, saving, or CLIP calls.
    torch.cuda.synchronize(job["device"])
    start = time.perf_counter()
    for step in range(job["steps"]):
        optimizer.zero_grad()
        if job["loss_type"] == "sds":
            loss = model.get_sds_loss(latents, embeddings, job["guidance_scale"])
        else:
            loss = model.get_sdi_loss(
                latents, embeddings, guidance_scale=job["guidance_scale"],
                current_iter=step, total_iters=job["steps"],
                inversion_guidance_scale=job["inversion_guidance_scale"],
                inversion_n_steps=job["inversion_n_steps"],
                inversion_eta=job["inversion_eta"], update_interval=job["update_interval"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_([latents], max_norm=1.0)
        optimizer.step()
        schedule.step()
    torch.cuda.synchronize(job["device"])
    elapsed = time.perf_counter() - start
    if not torch.isfinite(latents).all().item():
        raise RuntimeError("Non-finite optimized latent; this run will not be marked complete.")

    image, metadata, _ = artifacts(job)
    image.parent.mkdir(parents=True, exist_ok=True)
    temporary_image = image.with_suffix(".tmp.png")
    torch_to_pil(model.decode_latents(latents)).save(temporary_image)
    temporary_image.replace(image)
    write_json(metadata, {
        "fingerprint": job["fingerprint"], "config": job,
        "optimization_seconds": elapsed, "initial_latent_sha256": initial_hash,
        "image_sha256": digest(image), "gpu": torch.cuda.get_device_name(job["device"]),
        "torch_version": str(torch.__version__),
    })
    print(f"Saved {image}; optimization took {elapsed:.3f}s", flush=True)


def evaluate(jobs, output, device):
    import torch
    from eval import ClipEvaluator

    evaluator = ClipEvaluator(name="ViT-B/32").cuda(device)
    text_features = {p: evaluator.encode_text(p) for p in PROMPTS}
    results = {}
    initial_hashes = {}
    for index, job in enumerate(jobs, 1):
        image, metadata, _ = artifacts(job)
        data = json.loads(metadata.read_text())
        # Paired settings must use exactly the same initial latent.
        if job["seed"] is not None:
            pair = (job["prompt"], job["seed"])
            previous = initial_hashes.setdefault(pair, data["initial_latent_sha256"])
            if previous != data["initial_latent_sha256"]:
                raise RuntimeError(f"Initial latent mismatch for {pair}")
        if data.get("reused_existing"):
            score = data["clip_score"]
        else:
            score = evaluator(evaluator.encode_image(image), text_features[job["prompt"]]).item()
        if not math.isfinite(score):
            raise RuntimeError(f"Non-finite CLIP score for {image}")
        results[(job["group"], job["value"], job["seed"], job["prompt"])] = {
            "score": score, "optimization_seconds": data["optimization_seconds"],
            "image": str(image),
        }
        print(f"[CLIP {index}/{len(jobs)}] {score:.4f} {image}", flush=True)

    summaries = []
    for group, values in SWEEPS.items():
        for value in values:
            group_key, value_key = canonical(group, value)
            seed_scores, seed_times, seed_results = [], [], {}
            reused = (group_key, value_key, None, PROMPTS[0]) in results
            for seed in ((None,) if reused else SEEDS):
                seed_label = "existing_unseeded" if seed is None else f"seed_{seed}"
                folder = output / group / str(value) / seed_label
                folder.mkdir(parents=True, exist_ok=True)
                scores, times = [], []
                per_seed = {"source_dir": str(folder)}
                for prompt in PROMPTS:
                    result = results[(group_key, value_key, seed, prompt)]
                    name = Path(result["image"]).name
                    destination = folder / name
                    if destination.resolve() != Path(result["image"]).resolve():
                        shutil.copy2(result["image"], destination)
                    if seed is None or seed == SEEDS[0]:
                        figure = ROOT / "template/LaTeX/Figure/ablation" / group / str(value) / name
                        figure.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(result["image"], figure)
                    per_seed[str(destination)] = result["score"]
                    scores.append(result["score"])
                    if result["optimization_seconds"] is not None:
                        times.append(result["optimization_seconds"])
                seed_scores.append(statistics.mean(scores))
                if times:
                    seed_times.append(statistics.mean(times))
                per_seed["score"] = seed_scores[-1]
                per_seed["seed"] = seed
                write_json(folder / "eval.json", per_seed)
                seed_results[seed_label] = {
                    "clip_mean": seed_scores[-1],
                    "seconds_mean": statistics.mean(times) if times else None}
            summaries.append({
                "ablation": group, "value": value,
                "clip_mean": statistics.mean(seed_scores),
                "clip_std": statistics.stdev(seed_scores) if len(seed_scores) > 1 else None,
                "seconds_mean": statistics.mean(seed_times) if seed_times else None,
                "seconds_std": statistics.stdev(seed_times) if len(seed_times) > 1 else None,
                "reused_existing_unseeded": reused,
                "per_seed": seed_results,
            })
    write_json(output / "summary.json", {
        "prompts": PROMPTS, "seeds": SEEDS, "unique_runs": len(jobs),
        "reused_images": sum("reuse_source" in j for j in jobs),
        "aggregation": "Mean CLIP score over 3 prompts; one run per prompt and setting",
        "baseline_note": "Reused unseeded baselines use 3 existing prompts; seed std and runtime are unavailable.",
        "timing": "CUDA-synchronized optimization wall time, excluding loading/decoding/I/O/CLIP",
        "results": summaries,
    })
    columns = ("ablation", "value", "clip_mean", "clip_std", "seconds_mean", "seconds_std")
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    # A ready-to-use table is generated beside the results; existing .tex is untouched.
    lines = [r"\begin{table}[t]", r"  \centering\footnotesize",
             r"  \setlength{\tabcolsep}{4pt}",
             r"  \caption{\textbf{Ablation study.} Average CLIP score over three prompts.",
             r"  Each parameter is varied independently; existing baselines are reused.}",
             r"  \label{tab:ablation}", r"  \begin{tabular}{@{}lccccc@{}}", r"    \toprule"]
    titles = ("SDS: guidance scale", "SDI: inversion steps", "SDI: update interval")
    for (group, values), title in zip(SWEEPS.items(), titles):
        rows = [r for r in summaries if r["ablation"] == group]
        lines += [r"    \multicolumn{6}{@{}l}{\textit{" + title + r"}} \\",
                  "    Value & " + " & ".join(map(str, values)) + r" \\",
                  "    Mean & " + " & ".join(f'{r["clip_mean"]:.4f}' for r in rows) + r" \\",
                  r"    \midrule"]
    lines[-1] = r"    \bottomrule"
    lines += [r"  \end{tabular}", r"\end{table}"]
    (output / "table_2_ablation.tex").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=(
        "Run 3 independent sweeps x 5 values x 3 prompts = 45 comparisons, with one fixed seed. "
        "Deduplicate the shared SDI baseline and reuse existing saved results. "
        "Completed runs with matching configuration/source hashes are resumed automatically."))
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablation"))
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without loading models or writing outputs")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate seeded runs; saved baselines are still reused")
    parser.add_argument("--eval-only", action="store_true", help="Recompute CLIP and summaries from completed runs")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.worker.read_text()))
        return
    if args.device < 0:
        parser.error("--device must be nonnegative")
    if args.eval_only and args.overwrite:
        parser.error("--eval-only and --overwrite cannot be combined")
    output = args.output_dir.resolve()
    jobs = make_jobs(output, args.device)
    signature = source_signature()
    for job in jobs:
        job["implementation_signature"] = implementation_signature()
        job["fingerprint"] = hashlib.sha256(
            (signature + json.dumps(job, sort_keys=True)).encode()).hexdigest()
    print(f"Output: {output}\nPrompts: {PROMPTS}\nSeeds: {SEEDS}", flush=True)
    for group, values in SWEEPS.items():
        print(f"  {group}: {values}", flush=True)
    print("Logical comparisons: 3 sweeps x 5 values x 3 prompts = 45 (no seed repetitions)", flush=True)
    imported = sum("reuse_source" in j for j in jobs)
    print(f"Unique result images: {len(jobs)}; existing baseline images: {imported}", flush=True)
    pending = sum(not complete(j) for j in jobs if "reuse_source" not in j)
    print(f"New generations needed: {pending if not args.overwrite else len(jobs) - imported}", flush=True)
    if args.dry_run:
        print(f"Reusable completed runs: {sum(complete(j) for j in jobs)}")
        return
    import torch
    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {args.device} is unavailable; activate the lab environment and check GPU access.")
    if args.eval_only and not all(complete(j) for j in jobs if "reuse_source" not in j):
        raise RuntimeError("Some runs are missing, incomplete, or use different source/configuration hashes.")
    output.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        if "reuse_source" in job and not complete(job):
            import_baseline(job)
    write_json(output / "protocol.json", {
        "source_signature": signature, "prompts": PROMPTS, "seeds": SEEDS,
        "sweeps": SWEEPS, "jobs": jobs,
    })
    if not args.eval_only:
        for index, job in enumerate(jobs, 1):
            if source_signature() != signature:
                raise RuntimeError("Implementation changed during the experiment; stopped to avoid mixing versions.")
            if "reuse_source" in job or (not args.overwrite and complete(job)):
                print(f"[{index}/{len(jobs)}] resume: {artifacts(job)[0]}", flush=True)
                continue
            image, _, log = artifacts(job)
            image.parent.mkdir(parents=True, exist_ok=True)
            request = image.with_suffix(".request.json")
            write_json(request, job)
            print(f"[{index}/{len(jobs)}] {job['group']}={job['value']} seed={job['seed']} | {job['prompt']}", flush=True)
            env = dict(os.environ, PYTHONHASHSEED=str(job["seed"]))
            with log.open("w") as handle:
                result = subprocess.run([sys.executable, "-u", __file__, "--worker", str(request)],
                                        stdout=handle, stderr=subprocess.STDOUT, env=env)
            if result.returncode:
                tail = "\n".join(log.read_text(errors="replace").splitlines()[-25:])
                raise RuntimeError(f"Generation failed. Log: {log}\n{tail}")
            if not complete(job):
                raise RuntimeError(f"Run did not produce valid completion metadata: {image}")
    evaluate(jobs, output, args.device)
    print(f"Done: {output / 'summary.csv'}", flush=True)
    print("Seed 2024 images (existing images for unseeded baselines) copied to template/LaTeX/Figure/ablation/.")
    print("Existing .tex files unchanged.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)
PY
"${PYTHON:-python}" "$runner" "$@"
