#!/usr/bin/env python3
"""Assign a unique speaker profile (voice, age bracket, accent/dialect) to every
case across both datasets, so no two cases sound the same. Dimensions follow
established ASR/voice-agent fairness evaluation practice (Artie Bias Corpus's
age/gender/accent tagging; Koenecke et al. 2020's finding that African American
English carries the largest documented ASR disparity, hence its inclusion as its
own accent category rather than folding it into "American").

139 cases < 11 voices x 4 ages x 12 accents = 528 possible triples, so a
without-replacement sample guarantees every case gets a distinct combination.

Run:  python3 speaker_profiles.py
Writes: severe_failure_cases/speaker_manifest.json, test_cases/speaker_manifest.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Rough gender-presentation labels for OpenAI's 11 gpt-4o-mini-tts voices — a
# practical approximation (OpenAI does not officially label voices by gender),
# used only for reporting/coverage tracking, not for synthesis itself.
VOICES = {
    "alloy":   "neutral",
    "echo":    "male",
    "fable":   "male",
    "onyx":    "male",
    "nova":    "female",
    "shimmer": "female",
    "ash":     "male",
    "ballad":  "male",
    "coral":   "female",
    "sage":    "female",
    "verse":   "male",
}

AGE_BRACKETS = {
    "teen":         "a teenager, around 16 years old",
    "young_adult":  "a young adult in their mid-20s",
    "middle_aged":  "a middle-aged adult in their 40s",
    "older_adult":  "an older adult in their late 60s, with the slower, deliberate pacing common at that age",
}

# Accent/dialect set: broad regional coverage plus the specific dialects called
# out in ASR-bias literature as high-disparity (AAVE, Indian English, Nigerian
# English) rather than only "national" accents.
ACCENTS = {
    "general_american":  "a neutral General American",
    "southern_us":        "a Southern United States",
    "aave":                "an African American English (AAVE)",
    "british_rp":          "a British Received Pronunciation (Standard Southern British)",
    "scottish":            "a Scottish",
    "irish":               "an Irish",
    "indian_english":      "an Indian English",
    "nigerian_english":    "a Nigerian English",
    "australian":          "an Australian",
    "filipino_english":    "a Filipino English",
    "latino_english":      "a Latino/Spanish-accented English",
    "chinese_english":     "a Chinese-accented English",
}

SEED_SALT = "voice-guard-eval-speaker-profiles-v1"


def all_triples() -> list[tuple[str, str, str]]:
    return [(v, a, acc) for v in VOICES for a in AGE_BRACKETS for acc in ACCENTS]


def deterministic_shuffle(items: list, seed_key: str) -> list:
    """Stable, reproducible shuffle without relying on Python's random module
    seeding (which isn't guaranteed stable across versions) — sort by a hash of
    (seed_key, item) instead."""
    def keyfn(item):
        h = hashlib.sha1(f"{SEED_SALT}:{seed_key}:{item}".encode()).digest()
        return h
    return sorted(items, key=keyfn)


def assign_profiles(case_ids: list[str]) -> dict:
    triples = deterministic_shuffle(all_triples(), "triple-pool")
    case_ids_sorted = sorted(case_ids)  # stable input order regardless of filesystem listing order
    assert len(case_ids_sorted) <= len(triples), (
        f"{len(case_ids_sorted)} cases > {len(triples)} available unique profiles — "
        "expand a dimension (add voices/ages/accents) before proceeding."
    )
    manifest = {}
    for case_id, (voice, age, accent) in zip(case_ids_sorted, triples):
        manifest[case_id] = {
            "voice": voice,
            "gender_label": VOICES[voice],
            "age_bracket": age,
            "age_descriptor": AGE_BRACKETS[age],
            "accent": accent,
            "accent_descriptor": ACCENTS[accent],
        }
    return manifest


def build_instructions(profile: dict, acting_notes: str) -> str:
    return (
        f"Speak as {profile['age_descriptor']} speaker with {profile['accent_descriptor']} "
        f"accent. {acting_notes}"
    )


def process_dataset(dataset_dir: Path) -> dict:
    case_ids = sorted({p.name.rsplit("_", 1)[0] for p in dataset_dir.iterdir() if p.is_dir()})
    manifest = assign_profiles(case_ids)
    (dataset_dir / "speaker_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    from collections import Counter

    all_case_ids = []
    for name in ["severe_failure_cases", "test_cases"]:
        d = HERE / name
        ids = sorted({p.name.rsplit("_", 1)[0] for p in d.iterdir() if p.is_dir()})
        all_case_ids.extend(ids)

    print(f"Total cases across both datasets: {len(all_case_ids)}")
    print(f"Available unique (voice, age, accent) triples: {len(all_triples())}")

    # Assign across the COMBINED pool so uniqueness holds across both datasets,
    # not just within each one independently.
    manifest_all = assign_profiles(all_case_ids)

    seen_triples = set()
    for case_id, p in manifest_all.items():
        t = (p["voice"], p["age_bracket"], p["accent"])
        assert t not in seen_triples, f"duplicate profile triple for {case_id}: {t}"
        seen_triples.add(t)
    print(f"Verified {len(seen_triples)} unique triples across {len(manifest_all)} cases — no duplicates.")

    for name in ["severe_failure_cases", "test_cases"]:
        d = HERE / name
        ids = sorted({p.name.rsplit("_", 1)[0] for p in d.iterdir() if p.is_dir()})
        sub_manifest = {cid: manifest_all[cid] for cid in ids}
        (d / "speaker_manifest.json").write_text(json.dumps(sub_manifest, indent=2))
        print(f"\n{name}: wrote speaker_manifest.json for {len(sub_manifest)} cases")
        print("  gender coverage:", dict(Counter(p["gender_label"] for p in sub_manifest.values())))
        print("  age coverage:", dict(Counter(p["age_bracket"] for p in sub_manifest.values())))
        print("  accent coverage:", dict(Counter(p["accent"] for p in sub_manifest.values())))
