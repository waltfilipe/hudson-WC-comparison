import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extract_sofascore_match_events import (  # noqa: E402
    flatten_category_events,
    load_lineups_file,
    parse_match_id,
    players_from_lineups,
    PlayerInfo,
)


def test_parse_match_id_from_hash():
    url = "https://www.sofascore.com/football/match/argentina-austria/tUbsuWb#id:15186502"
    assert parse_match_id(url) == 15186502
    assert parse_match_id("15186502") == 15186502


def test_players_from_lineups():
    lineups = json.loads((ROOT / "tests/fixtures/sofascore_lineups_sample.json").read_text())
    players = players_from_lineups(lineups)
    assert len(players) == 2
    assert players[0].player_name == "Lionel Messi"


def test_flatten_passes_and_carries():
    payload = json.loads((ROOT / "tests/fixtures/sofascore_rating_breakdown_sample.json").read_text())
    player = PlayerInfo(
        player_id=12994,
        player_name="Lionel Messi",
        short_name="L. Messi",
        position="F",
        team="Argentina",
        side="home",
        shirt_number=10,
        substitute=False,
    )

    passes = flatten_category_events(payload["passes"], "passes", player, 15186502)
    carries = flatten_category_events(payload["ball-carries"], "ball-carries", player, 15186502)

    assert len(passes) == 2
    assert passes[0]["x"] == 72.0
    assert passes[0]["end_x"] == 68.0
    assert passes[0]["outcome"] == "successful"

    assert len(carries) == 1
    assert carries[0]["end_x"] == 75.0
    assert carries[0]["progressive"] is True

    all_df = pd.DataFrame(passes + carries)
    assert set(all_df["category"]) == {"passes", "ball-carries"}


def test_load_lineups_file():
    lineups = load_lineups_file(ROOT / "tests/fixtures/sofascore_lineups_sample.json")
    assert "home" in lineups and "away" in lineups
    assert len(players_from_lineups(lineups)) == 2


if __name__ == "__main__":
    test_parse_match_id_from_hash()
    test_players_from_lineups()
    test_flatten_passes_and_carries()
    test_load_lineups_file()
    print("ok")
