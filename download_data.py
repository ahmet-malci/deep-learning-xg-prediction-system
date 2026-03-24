import json
import random
from pathlib import Path

import requests
from tqdm import tqdm

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
DATA = Path("data")
MATCHES_DIR = DATA / "matches"
EVENTS_DIR = DATA / "events"
PROCESSED_DIR = DATA / "processed"

MATCHES_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


TARGET_MATCHES = 300
TARGET_SHOTS = 8000


NUM_COMPETITIONS = 8
MIN_MATCHES_REQUIRED = 50
EXCLUDE_PENALTIES = True

def get_json(url: str):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

def is_womens_comp(comp: dict) -> bool:
    cname = (comp.get("competition_name") or "").lower()
    sname = (comp.get("season_name") or "").lower()
    gender = str(comp.get("competition_gender", "")).lower()

    if "women" in cname or "women's" in cname or "womens" in cname:
        return True
    if "women" in sname or "women's" in sname or "womens" in sname:
        return True
    if gender in {"female", "women", "womens"}:
        return True
    return False

def count_shots_in_events(events: list) -> int:
    cnt = 0
    for ev in events:
        if ev.get("type", {}).get("name") != "Shot":
            continue
        shot = ev.get("shot", {})
        outcome = shot.get("outcome", {}).get("name")
        loc = ev.get("location")
        if outcome is None or not loc:
            continue
        if EXCLUDE_PENALTIES:
            shot_type = shot.get("type", {}).get("name")
            if shot_type == "Penalty":
                continue
        cnt += 1
    return cnt

def main():
    comps = get_json(f"{BASE}/competitions.json")
    save_json(comps, DATA / "competitions.json")
    print("competitions.json hazır. Toplam kayıt:", len(comps))

    chosen = []
    used_names = set()

    for comp in comps:
        cname = comp.get("competition_name")
        cid = comp.get("competition_id")
        sid = comp.get("season_id")
        sname = comp.get("season_name")

        if not (cname and cid is not None and sid is not None):
            continue
        if is_womens_comp(comp):
            continue
        if cname in used_names:
            continue

        matches = get_json(f"{BASE}/matches/{cid}/{sid}.json")
        if len(matches) < MIN_MATCHES_REQUIRED:
            continue

        chosen.append((cid, sid, cname, sname, matches))
        used_names.add(cname)


        save_json(matches, MATCHES_DIR / f"{cid}_{sid}.json")

        if len(chosen) >= NUM_COMPETITIONS:
            break

    if not chosen:
        raise RuntimeError("Uygun lig/sezon bulunamadı.")

    print("\nSeçilen lig/sezon havuzu:")
    for cid, sid, cname, sname, matches in chosen:
        print(f"- {cname} | {sname} -> {len(matches)} maç (competition_id={cid}, season_id={sid})")

    pools = []
    for cid, sid, cname, sname, matches in chosen:
        ids = [m["match_id"] for m in matches if "match_id" in m]
        random.shuffle(ids)
        pools.append({
            "competition_id": cid,
            "season_id": sid,
            "competition_name": cname,
            "season_name": sname,
            "match_ids": ids,
            "cursor": 0
        })

    total_matches = 0
    total_shots = 0
    selected_matches = []

    pbar = tqdm(total=TARGET_MATCHES, desc="Downloading matches/events")
    while total_matches < TARGET_MATCHES or total_shots < TARGET_SHOTS:
        progressed = False

        for pool in pools:
            if total_matches >= TARGET_MATCHES and total_shots >= TARGET_SHOTS:
                break

            if pool["cursor"] >= len(pool["match_ids"]):
                continue

            match_id = pool["match_ids"][pool["cursor"]]
            pool["cursor"] += 1
            progressed = True

            event_path = EVENTS_DIR / f"{match_id}.json"

            if event_path.exists():
                events = json.loads(event_path.read_text(encoding="utf-8"))
            else:
                events = get_json(f"{BASE}/events/{match_id}.json")
                save_json(events, event_path)

            shot_count = count_shots_in_events(events)

            total_matches += 1
            total_shots += shot_count
            selected_matches.append({
                "match_id": match_id,
                "competition_name": pool["competition_name"],
                "season_name": pool["season_name"],
                "competition_id": pool["competition_id"],
                "season_id": pool["season_id"],
                "shots_in_match": shot_count
            })

            pbar.update(1)
            pbar.set_postfix({"shots": total_shots})


            if total_matches >= TARGET_MATCHES and total_shots >= TARGET_SHOTS:
                break

        if not progressed:
            break

    pbar.close()

    manifest = {
        "target_matches": TARGET_MATCHES,
        "target_shots": TARGET_SHOTS,
        "exclude_penalties": EXCLUDE_PENALTIES,
        "selected_total_matches": total_matches,
        "selected_total_shots": total_shots,
        "selected_matches": selected_matches,
        "chosen_competitions": [
            {"competition_id": cid, "season_id": sid, "competition_name": cname, "season_name": sname}
            for cid, sid, cname, sname, _ in chosen
        ]
    }
    save_json(manifest, PROCESSED_DIR / "selection_manifest.json")

    print("\nİndirme tamamlandı.")
    print("Toplam maç:", total_matches)
    print("Toplam şut (penaltı hariç):", total_shots)
    print("Manifest:", (PROCESSED_DIR / "selection_manifest.json").resolve())

    if total_matches < TARGET_MATCHES or total_shots < TARGET_SHOTS:
        print("\nHedefler tam karşılanamadı.")


if __name__ == "__main__":
    main()
