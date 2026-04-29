#!/usr/bin/env python3
"""
Lyrics transcription evaluation — WhisperX pipeline.

Metrics:
  WER   — word error rate          (primary text quality)
  CER   — character error rate     (better for JP, RU, non-latin)
  TSmae — timestamp MAE in seconds (line-level sync quality)

Usage:
  python eval.py --manifest manifest.json --audio-dir ./audio --device cuda
  python eval.py --manifest manifest.json --audio-dir ./audio --device cpu --compute-type int8
"""

import os
import re
import json
import argparse
import unicodedata
from pathlib import Path
from typing import Optional
from difflib import SequenceMatcher

import numpy as np
import jiwer
import whisperx


# ─── Text normalisation ───────────────────────────────────────────────────────

# Languages that don't use spaces between words — use CER as primary metric
CHAR_LANGS = {"ja", "zh", "ko"}

def normalize(text: str, lang: str = "en") -> str:
    """Lowercase, unicode NFC, strip punctuation, collapse spaces."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFC", text)
    # Remove punctuation (keep letters, digits, spaces)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── LRC parsing ─────────────────────────────────────────────────────────────

LRC_RE = re.compile(r"\[(\d{1,2}):(\d{2}\.\d+)\](.*)")

def parse_lrc(lrc: str) -> list[dict]:
    """
    Parse standard LRC into [{time_s: float, text: str}].
    Ignores metadata tags ([ar:], [ti:], etc.).
    """
    lines = []
    for m in LRC_RE.finditer(lrc):
        mins, secs, text = int(m.group(1)), float(m.group(2)), m.group(3).strip()
        if text:
            lines.append({"time": mins * 60 + secs, "text": text})
    return sorted(lines, key=lambda x: x["time"])


# ─── Metrics ─────────────────────────────────────────────────────────────────

def _prep(text: str) -> str:
    """Lowercase + strip punctuation перед передачей в jiwer."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def compute_wer(hypothesis: str, reference: str) -> float:
    if not reference.strip() or not hypothesis.strip():
        return 1.0
    return jiwer.wer(_prep(reference), _prep(hypothesis))


def compute_cer(hypothesis: str, reference: str) -> float:
    if not reference.strip() or not hypothesis.strip():
        return 1.0
    return jiwer.cer(_prep(reference), _prep(hypothesis))


def compute_timestamp_mae(
    pred_lines: list[dict],
    ref_lines: list[dict],
    pred_words: list[dict] = None,
    time_window: float = 3.0,
    min_text_sim: float = 0.4,
) -> Optional[float]:
    """
    Time-anchored word matching.

    Для каждой LRC строки:
    1. Берём pred слова в окне ±time_window секунд от LRC таймстемпа
    2. Среди них ищем лучшее текстовое совпадение с первыми словами строки
    3. Если совпадение >= min_text_sim — считаем ошибку

    Это надёжнее чем чисто текстовый матчинг:
    "that" в 34s не может сматчиться с LRC строкой в 10s.
    Работает даже когда Whisper цензурит мат или пишет иначе.
    """
    if not ref_lines or not pred_words:
        return None

    errors = []
    matched = 0

    for ref_line in ref_lines:
        ref_t = ref_line["time"]
        ref_tokens = [w for w in normalize(ref_line["text"]).split() if len(w) > 2]
        if not ref_tokens:
            continue
        anchor = " ".join(ref_tokens[:2]) if len(ref_tokens) >= 2 else ref_tokens[0]

        # Кандидаты: pred слова в окне ±time_window от ref таймстемпа
        candidates = [
            w for w in pred_words
            if abs(w["start"] - ref_t) <= time_window
        ]
        if not candidates:
            continue

        # Лучший текстовый матч среди кандидатов
        best_score, best_word = 0.0, None
        for i, pw in enumerate(candidates):
            pw_norm = normalize(pw["word"])
            # Окно из 2 слов для сравнения
            next_pw = candidates[i+1]["word"] if i+1 < len(candidates) else ""
            window_text = (pw_norm + " " + normalize(next_pw)).strip()
            score = SequenceMatcher(None, anchor, window_text).ratio()
            if score > best_score:
                best_score, best_word = score, pw

        if best_word and best_score >= min_text_sim:
            err = abs(best_word["start"] - ref_t)
            errors.append(err)
            matched += 1

    print(f"   [ts] matched {matched}/{len(ref_lines)} ref lines via time+text alignment")
    if not errors:
        return None
    return float(np.mean(errors))


def detect_hallucination(segments: list[dict], repeat_thresh: int = 3) -> bool:
    """
    Heuristic: Whisper sometimes loops a phrase on silence.
    Flag if any segment text repeats >N times in a row.
    """
    texts = [s.get("text", "").strip().lower() for s in segments]
    for i in range(len(texts) - repeat_thresh):
        window = texts[i : i + repeat_thresh]
        if len(set(window)) == 1 and window[0]:
            return True
    return False


# ─── WhisperX transcription ───────────────────────────────────────────────────

def transcribe(
    audio_path: str,
    language: Optional[str],
    device: str,
    compute_type: str,
) -> dict:
    """
    Run WhisperX large-v3 with forced word alignment.

    Returns:
      text       — full transcript string
      words      — [{word, start, end, score}]
      lines      — [{time, text}]  (segment-level, maps to LRC lines)
      language   — detected language code
      hallucination — bool flag
    """
    model = whisperx.load_model(
        "large-v3", device,
        compute_type=compute_type,
        language=language,
    )
    audio = whisperx.load_audio(audio_path)

    result = model.transcribe(audio, batch_size=16)
    detected_lang = result.get("language", language or "en")

    is_hallucination = detect_hallucination(result["segments"])

    # Forced phoneme alignment for word-level timestamps
    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=detected_lang, device=device
        )
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, device,
            return_char_alignments=False,
        )
    except Exception as e:
        print(f"  [warn] alignment failed ({e}), using segment-level timestamps only")

    words = []
    lines = []
    for seg in result["segments"]:
        line_text = seg["text"].strip()
        if not line_text:
            continue
        lines.append({"time": seg["start"], "text": line_text})
        for w in seg.get("words", []):
            if "start" in w:
                words.append({
                    "word": w["word"].strip(),
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                    "score": round(w.get("score", 0.0), 3),
                })

    full_text = " ".join(l["text"] for l in lines)
    return {
        "text": full_text,
        "words": words,
        "lines": lines,
        "language": detected_lang,
        "hallucination": is_hallucination,
    }


# ─── Ground truth fetching ────────────────────────────────────────────────────

def load_or_fetch_gt(entry: dict, cache_dir: Path) -> tuple[str, list[dict]]:
    """
    Returns (plain_text, lrc_lines).
    Tries cache → syncedlyrics → lrclib.net → empty.
    """
    artist, title = entry["artist"], entry["title"]
    slug = re.sub(r"[^\w]", "_", f"{artist}_{title}").lower()
    cache_lrc = cache_dir / f"{slug}.lrc"
    cache_txt = cache_dir / f"{slug}.txt"

    # --- cache hit ---
    if cache_lrc.exists():
        raw = cache_lrc.read_text(encoding="utf-8")
        lines = parse_lrc(raw)
        return " ".join(l["text"] for l in lines), lines

    if cache_txt.exists():
        return cache_txt.read_text(encoding="utf-8"), []

    # --- syncedlyrics ---
    try:
        import syncedlyrics
        lrc = syncedlyrics.search(f"{artist} {title}")
        # Убеждаемся что это именно LRC с таймстемпами, а не plain text
        if lrc and LRC_RE.search(lrc):
            cache_lrc.write_text(lrc, encoding="utf-8")
            lines = parse_lrc(lrc)
            return " ".join(l["text"] for l in lines), lines
    except Exception as e:
        print(f"  [warn] syncedlyrics: {e}")

    # --- lrclib.net fallback ---
    try:
        import urllib.request, urllib.parse
        q = urllib.parse.urlencode({"artist_name": artist, "track_name": title})
        url = f"https://lrclib.net/api/search?{q}"
        req = urllib.request.Request(url, headers={"Lrclib-Client": "lyrics-eval/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        if data:
            synced = data[0].get("syncedLyrics") or ""
            plain = data[0].get("plainLyrics") or ""
            if synced:
                cache_lrc.write_text(synced, encoding="utf-8")
                lines = parse_lrc(synced)
                return " ".join(l["text"] for l in lines), lines
            if plain:
                cache_txt.write_text(plain, encoding="utf-8")
                return plain, []
    except Exception as e:
        print(f"  [warn] lrclib: {e}")

    return "", []


# ─── Main ─────────────────────────────────────────────────────────────────────

def evaluate(
    manifest_path: str,
    audio_dir: str,
    device: str,
    compute_type: str,
    out_path: str = "eval_results.json",
):
    manifest = json.loads(Path(manifest_path).read_text())
    audio_dir = Path(audio_dir)
    cache_dir = Path("lyrics_cache")
    cache_dir.mkdir(exist_ok=True)

    results = []

    for entry in manifest:
        audio_file = audio_dir / entry["file"]
        artist = entry["artist"]
        title = entry["title"]
        lang = entry.get("language")  # None = auto-detect

        print(f"\n{'─'*60}")
        print(f"▶  {artist} — {title}  [{lang or 'auto'}]")

        if not audio_file.exists():
            print(f"   [skip] file not found: {audio_file}")
            continue

        # 1. Ground truth
        ref_text, ref_lines = load_or_fetch_gt(entry, cache_dir)
        has_ref = bool(ref_text.strip())
        has_ts  = bool(ref_lines)
        print(f"   GT: {'text+timestamps' if has_ts else 'text only' if has_ref else 'NOT FOUND'}")

        # 2. Transcription
        pred = transcribe(str(audio_file), lang, device, compute_type)
        print(f"   Detected lang: {pred['language']}  |  words: {len(pred['words'])}  |  hallucination: {pred['hallucination']}")

        row = {
            "file": entry["file"],
            "artist": artist,
            "title": title,
            "language": pred["language"],
            "hallucination": pred["hallucination"],
            "wer": None,
            "cer": None,
            "timestamp_mae_s": None,
            "n_words_pred": len(pred["words"]),
        }

        if has_ref and not pred["hallucination"]:
            use_cer_primary = pred["language"] in CHAR_LANGS
            row["wer"] = round(compute_wer(pred["text"], ref_text), 4)
            row["cer"] = round(compute_cer(pred["text"], ref_text), 4)

            primary_metric = "CER" if use_cer_primary else "WER"
            primary_val = row["cer"] if use_cer_primary else row["wer"]
            print(f"   {primary_metric}={primary_val:.3f}  WER={row['wer']:.3f}  CER={row['cer']:.3f}")

            if has_ts:
                mae = compute_timestamp_mae(pred["lines"], ref_lines, pred_words=pred["words"])
                row["timestamp_mae_s"] = round(mae, 3) if mae is not None else None
                print(f"   Timestamp MAE={row['timestamp_mae_s']}s  "
                      f"(matched {len(pred['lines'])} pred vs {len(ref_lines)} ref lines)")
        elif pred["hallucination"]:
            print("   [skip metrics] hallucination detected")
        else:
            print("   [skip metrics] no reference found")

        results.append(row)

    # ── Summary ───────────────────────────────────────────────────────────────
    wers = [r["wer"] for r in results if r["wer"] is not None]
    cers = [r["cer"] for r in results if r["cer"] is not None]
    maes = [r["timestamp_mae_s"] for r in results if r["timestamp_mae_s"] is not None]
    hallucinated = sum(1 for r in results if r["hallucination"])

    print(f"\n{'═'*60}")
    print(f"SUMMARY  {len(results)} tracks  |  {len(wers)} with text GT  |  {len(maes)} with timestamp GT")
    if wers:
        print(f"  avg WER:           {np.mean(wers):.3f}  (median {np.median(wers):.3f}, best {min(wers):.3f})")
    if cers:
        print(f"  avg CER:           {np.mean(cers):.3f}  (median {np.median(cers):.3f})")
    if maes:
        print(f"  avg Timestamp MAE: {np.mean(maes):.2f}s  (median {np.median(maes):.2f}s)")
    print(f"  hallucinations:    {hallucinated}/{len(results)}")

    output = {
        "summary": {
            "n_tracks": len(results),
            "avg_wer": round(float(np.mean(wers)), 4) if wers else None,
            "median_wer": round(float(np.median(wers)), 4) if wers else None,
            "avg_cer": round(float(np.mean(cers)), 4) if cers else None,
            "avg_timestamp_mae_s": round(float(np.mean(maes)), 3) if maes else None,
            "median_timestamp_mae_s": round(float(np.median(maes)), 3) if maes else None,
            "hallucination_rate": round(hallucinated / len(results), 3) if results else None,
        },
        "tracks": results,
    }
    Path(out_path).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nResults → {out_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval lyrics transcription quality")
    parser.add_argument("--manifest",     default="manifest.json")
    parser.add_argument("--audio-dir",    default="./audio")
    parser.add_argument("--out",          default="eval_results.json")
    parser.add_argument("--device",       default="cuda",
                        help="cuda | cpu | mps")
    parser.add_argument("--compute-type", default="float16",
                        choices=["float16", "int8", "float32"],
                        help="float16 for GPU, int8 for CPU")
    args = parser.parse_args()

    evaluate(args.manifest, args.audio_dir, args.device, args.compute_type, args.out)