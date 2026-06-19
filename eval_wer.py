#!/usr/bin/env python3
"""
WER evaluation for MedASR
=========================

Measure transcription accuracy so it can be tracked, not guessed. Computes
Word Error Rate (WER) of MedASR against reference transcripts, with a
substitution/deletion/insertion breakdown and the most common confusions.

Reference data — pick one:
  1) JSONL manifest, one object per line:
        {"audio": "clips/a.wav", "reference": "the patient reports chest pain"}
     python eval_wer.py --manifest data/eval.jsonl
  2) A folder where each `x.wav` has a sibling `x.txt` reference:
        python eval_wer.py --dir data/eval/
  3) Self-contained demo (synthesizes known-reference audio via macOS `say`):
        python eval_wer.py --demo

Notes:
  - Text is normalized before scoring: MedASR's {period}/{comma}/... markers are
    dropped, then lowercased, punctuation stripped, whitespace collapsed.
  - WER = (Substitutions + Deletions + Insertions) / words_in_reference.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent / "backend"))

_MARKERS = re.compile(
    r"\{period\}|\{comma\}|\{colon\}|\{new paragraph\}|\{open_bracket\}|"
    r"\{close_bracket\}|</s>"
)
_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: str) -> List[str]:
    """Lowercase, drop ASR markers + punctuation, collapse whitespace → words."""
    text = _MARKERS.sub(" ", text)
    text = _PUNCT.sub(" ", text.lower())
    return text.split()


def align(ref: List[str], hyp: List[str]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """Word-level Levenshtein alignment. Returns ops: ('='|'S'|'D'|'I', ref, hyp)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    ops: List[Tuple[str, Optional[str], Optional[str]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0 and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1)
        ):
            tag = "=" if ref[i - 1] == hyp[j - 1] else "S"
            ops.append((tag, ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def score(ref_text: str, hyp_text: str) -> Dict:
    ref, hyp = normalize(ref_text), normalize(hyp_text)
    ops = align(ref, hyp)
    s = sum(1 for o in ops if o[0] == "S")
    d = sum(1 for o in ops if o[0] == "D")
    ins = sum(1 for o in ops if o[0] == "I")
    n = max(len(ref), 1)
    return {
        "wer": (s + d + ins) / n,
        "S": s, "D": d, "I": ins, "N": len(ref),
        "subs": [(o[1], o[2]) for o in ops if o[0] == "S"],
    }


# --------------------------------------------------------------- data loading
def load_manifest(path: Path) -> List[Dict]:
    items = []
    base = path.parent
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        audio = Path(obj["audio"])
        if not audio.is_absolute():
            audio = base / audio
        items.append({"audio": str(audio), "reference": obj["reference"]})
    return items


def load_dir(path: Path) -> List[Dict]:
    items = []
    for wav in sorted(path.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if txt.exists():
            items.append(
                {"audio": str(wav), "reference": txt.read_text(encoding="utf-8").strip()}
            )
    return items


_DEMO_SENTENCES = [
    "The patient reports chest pain radiating to the left arm.",
    "Blood pressure is elevated and the heart rhythm is irregular.",
    "Auscultation reveals a systolic murmur at the apex.",
]


def load_demo() -> List[Dict]:
    if not Path("/usr/bin/say").exists():
        print("`say` not available (macOS only); provide --manifest or --dir.")
        sys.exit(1)
    tmp = Path(tempfile.mkdtemp(prefix="wer_demo_"))
    items = []
    for idx, sentence in enumerate(_DEMO_SENTENCES):
        aiff = tmp / f"s{idx}.aiff"
        subprocess.run(
            ["say", "-v", "Samantha", "-r", "170", "-o", str(aiff), sentence],
            check=True,
        )
        items.append({"audio": str(aiff), "reference": sentence})
    print(f"Synthesized {len(items)} demo clips via `say` in {tmp}")
    return items


# ----------------------------------------------------------------------- main
def evaluate(mgr, items: List[Dict], verbose: bool = True, show_subs: int = 12) -> float:
    tot = {"S": 0, "D": 0, "I": 0, "N": 0}
    all_subs: Counter = Counter()
    conf_sum, conf_n = 0.0, 0

    if verbose:
        print(f"\n{'file':<28}{'WER':>8}{'conf':>8}{'S':>5}{'D':>5}{'I':>5}{'N':>6}")
        print("-" * 68)
    for it in items:
        result = mgr.medasr.transcribe_file(it["audio"])
        sc = score(it["reference"], result.text)
        for k in ("S", "D", "I", "N"):
            tot[k] += sc[k]
        all_subs.update(sc["subs"])
        conf_sum += result.confidence
        conf_n += 1
        if verbose:
            name = Path(it["audio"]).name[:26]
            print(
                f"{name:<28}{sc['wer']*100:>7.1f}%{result.confidence*100:>7.0f}%"
                f"{sc['S']:>5}{sc['D']:>5}{sc['I']:>5}{sc['N']:>6}"
            )

    n = max(tot["N"], 1)
    agg = (tot["S"] + tot["D"] + tot["I"]) / n
    if verbose:
        print("-" * 68)
        print(
            f"{'AGGREGATE':<28}{agg*100:>7.1f}%{(conf_sum/max(conf_n,1))*100:>7.0f}%"
            f"{tot['S']:>5}{tot['D']:>5}{tot['I']:>5}{n:>6}"
        )
        print(f"\nWER = {agg*100:.2f}%  over {n} reference words, {conf_n} clips")
        if all_subs:
            print("\nTop substitutions (reference → hypothesis):")
            for (r, h), c in all_subs.most_common(show_subs):
                print(f"  {c:>3}×  {r}  →  {h}")
    return agg


def run(items: List[Dict], compare: bool = False):
    from unified_model_manager import UnifiedModelManager

    print("Loading MedASR…")
    mgr = UnifiedModelManager()
    if not mgr.medasr.load():
        print("MedASR failed to load")
        sys.exit(1)

    if not compare:
        evaluate(mgr, items)
        return

    print("\n=== preprocessing OFF ===")
    mgr.medasr.preprocess_audio = False
    wer_off = evaluate(mgr, items, verbose=True, show_subs=0)
    print("\n=== preprocessing ON ===")
    mgr.medasr.preprocess_audio = True
    wer_on = evaluate(mgr, items, verbose=True, show_subs=0)

    delta = (wer_off - wer_on) * 100
    print("\n" + "=" * 40)
    print(f"  WER  off={wer_off*100:.2f}%   on={wer_on*100:.2f}%")
    verdict = "improves" if delta > 0 else ("hurts" if delta < 0 else "no change")
    print(f"  preprocessing {verdict} WER by {abs(delta):.2f} pts")
    print("=" * 40)


def main():
    ap = argparse.ArgumentParser(
        description="Measure MedASR word error rate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", help="JSONL with {audio, reference} per line")
    g.add_argument("--dir", help="folder of x.wav + x.txt reference pairs")
    g.add_argument("--demo", action="store_true", help="synthesize demo audio via macOS say")
    ap.add_argument(
        "--compare", action="store_true",
        help="run with audio preprocessing OFF then ON and report the WER delta",
    )
    args = ap.parse_args()

    if args.demo:
        items = load_demo()
    elif args.manifest:
        items = load_manifest(Path(args.manifest))
    else:
        items = load_dir(Path(args.dir))

    if not items:
        print("No (audio, reference) pairs found.")
        sys.exit(1)
    run(items, compare=args.compare)


if __name__ == "__main__":
    main()
