#!/usr/bin/env python3
"""Extrai passes e conduções (ball-carries) de todos os jogadores de uma partida SofaScore.

Exemplo:
    python extract_sofascore_match_events.py \\
        "https://www.sofascore.com/football/match/argentina-austria/tUbsuWb#id:15186502"

Saída (pasta output/ por padrão):
    - lineups.csv
    - passes.csv
    - ball_carries.csv
    - all_events.csv
    - raw/  (JSON bruto por jogador, opcional)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import pandas as pd

DEFAULT_MATCH_URL = (
    "https://www.sofascore.com/football/match/argentina-austria/tUbsuWb#id:15186502"
)
EVENT_CATEGORIES = ("passes", "ball-carries")
DEFAULT_IMPERSONATE = ("chrome131", "chrome124", "chrome120", "safari184", "edge101")


@dataclass(frozen=True)
class PlayerInfo:
    player_id: int
    player_name: str
    short_name: str
    position: str
    team: str
    side: str
    shirt_number: str | int | None
    substitute: bool


def parse_match_id(match_url_or_id: str) -> int:
    text = str(match_url_or_id).strip()
    if text.isdigit():
        return int(text)

    if "#id:" in text:
        return int(text.rsplit("#id:", maxsplit=1)[-1].split("?")[0])

    path_tail = urlparse(text).path.rstrip("/").split("/")[-1]
    if path_tail.isdigit():
        return int(path_tail)

    raise ValueError(
        "Não foi possível obter o match id. Use a URL com '#id:123456' ou passe --match-id."
    )


def flatten_category_events(
    events: list[dict[str, Any]],
    category: str,
    player: PlayerInfo,
    match_id: int,
) -> list[dict[str, Any]]:
    if not events:
        return []

    frame = pd.json_normalize(events, sep=".")
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        row = {
            "match_id": match_id,
            "category": category,
            "player_id": player.player_id,
            "player_name": player.player_name,
            "player_short_name": player.short_name,
            "player_position": player.position,
            "team": player.team,
            "side": player.side,
            "shirt_number": player.shirt_number,
            "substitute": player.substitute,
        }
        row.update(record)

        row["x"] = _first_present(record, "playerCoordinates.x", "draw.start.x")
        row["y"] = _first_present(record, "playerCoordinates.y", "draw.start.y")
        row["end_x"] = _first_present(
            record,
            "passEndCoordinates.x",
            "carryEndCoordinates.x",
            "endCoordinates.x",
            "draw.end.x",
        )
        row["end_y"] = _first_present(
            record,
            "passEndCoordinates.y",
            "carryEndCoordinates.y",
            "endCoordinates.y",
            "draw.end.y",
        )
        rows.append(row)
    return rows


def _first_present(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def players_from_lineups(lineups: dict[str, Any], team_names: dict[str, str] | None = None) -> list[PlayerInfo]:
    team_names = team_names or {}
    players: list[PlayerInfo] = []

    for side in ("home", "away"):
        team_block = lineups.get(side)
        if not team_block:
            continue
        team_name = team_names.get(side) or team_block.get("team", {}).get("name") or side
        for item in team_block.get("players", []):
            player = item.get("player", {})
            players.append(
                PlayerInfo(
                    player_id=int(player["id"]),
                    player_name=str(player.get("name", "")),
                    short_name=str(player.get("shortName", "")),
                    position=str(player.get("position", "")),
                    team=str(team_name),
                    side=side,
                    shirt_number=item.get("shirtNumber", item.get("jerseyNumber")),
                    substitute=bool(item.get("substitute", False)),
                )
            )
    return players


class CurlCffiClient:
    def __init__(self, match_url: str, match_id: int, delay: float = 0.8) -> None:
        from curl_cffi import requests

        self.match_id = match_id
        self.match_url = match_url
        self.delay = delay
        self.session = requests.Session()
        self._requests = requests
        self._warm_up()

    def _warm_up(self) -> None:
        self.session.get(
            self.match_url,
            headers={"Accept-Language": "en-US,en;q=0.9"},
            impersonate=DEFAULT_IMPERSONATE[0],
            timeout=30,
        )
        time.sleep(self.delay)

    def _headers(self) -> dict[str, str]:
        return {
            "Referer": self.match_url,
            "Origin": "https://www.sofascore.com",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def get_json(self, path: str) -> dict[str, Any]:
        url = f"{API_BASE}/{path.lstrip('/')}"
        last_error: Exception | None = None

        for impersonate in DEFAULT_IMPERSONATE:
            try:
                response = self.session.get(
                    url,
                    headers=self._headers(),
                    impersonate=impersonate,
                    timeout=30,
                )
                if response.status_code == 403:
                    last_error = RuntimeError(f"403 Forbidden em {url} ({impersonate})")
                    continue
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict) and payload.get("error"):
                    raise RuntimeError(f"Erro da API em {url}: {payload['error']}")
                time.sleep(self.delay)
                return payload
            except Exception as exc:  # noqa: BLE001 - queremos tentar todos os impersonates
                last_error = exc

        raise RuntimeError(f"Falha ao acessar {url}") from last_error

    def get_lineups(self) -> dict[str, Any]:
        return self.get_json(f"event/{self.match_id}/lineups")

    def get_event(self) -> dict[str, Any]:
        return self.get_json(f"event/{self.match_id}")

    def get_rating_breakdown(self, player_id: int) -> dict[str, Any]:
        return self.get_json(f"event/{self.match_id}/player/{player_id}/rating-breakdown")


class BrowserClient:
    """Fallback com Chrome real (undetected-chromedriver), útil quando curl_cffi recebe 403."""

    def __init__(self, match_url: str, match_id: int, delay: float = 1.5) -> None:
        import undetected_chromedriver as uc
        from bs4 import BeautifulSoup

        self.match_id = match_id
        self.match_url = match_url
        self.delay = delay
        self._BeautifulSoup = BeautifulSoup

        options = uc.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")

        self.driver = uc.Chrome(options=options)
        self.driver.get(self.match_url)
        time.sleep(self.delay)

    def close(self) -> None:
        self.driver.quit()

    def get_json(self, path: str) -> dict[str, Any]:
        url = f"{API_BASE}/{path.lstrip('/')}"
        self.driver.get(url)
        time.sleep(self.delay)
        text = self._BeautifulSoup(self.driver.page_source, "html.parser").get_text()
        payload = json.loads(text)
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(f"Erro da API em {url}: {payload['error']}")
        return payload

    def get_lineups(self) -> dict[str, Any]:
        return self.get_json(f"event/{self.match_id}/lineups")

    def get_event(self) -> dict[str, Any]:
        return self.get_json(f"event/{self.match_id}")

    def get_rating_breakdown(self, player_id: int) -> dict[str, Any]:
        return self.get_json(f"event/{self.match_id}/player/{player_id}/rating-breakdown")


def build_client(mode: str, match_url: str, match_id: int, delay: float):
    if mode == "browser":
        return BrowserClient(match_url, match_id, delay=delay)

    try:
        return CurlCffiClient(match_url, match_id, delay=delay)
    except Exception:
        if mode == "curl":
            raise
        return BrowserClient(match_url, match_id, delay=delay)


def team_names_from_event(event_payload: dict[str, Any]) -> dict[str, str]:
    event = event_payload.get("event", event_payload)
    return {
        "home": event.get("homeTeam", {}).get("name", "home"),
        "away": event.get("awayTeam", {}).get("name", "away"),
    }


def extract_match_events(
    match_url_or_id: str,
    output_dir: Path,
    *,
    mode: str = "auto",
    delay: float = 0.8,
    save_raw: bool = False,
    lineups_file: Path | None = None,
    categories: Iterable[str] = EVENT_CATEGORIES,
) -> dict[str, pd.DataFrame]:
    match_id = parse_match_id(match_url_or_id)
    match_url = match_url_or_id if "sofascore.com" in str(match_url_or_id) else (
        f"https://www.sofascore.com/football/match/match/{match_id}#id:{match_id}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    if save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    client = None
    browser_client = isinstance(mode, str) and mode == "browser"

    try:
        if lineups_file:
            lineups = json.loads(lineups_file.read_text(encoding="utf-8"))
            team_names = {}
        else:
            client = build_client(mode, match_url, match_id, delay=delay)
            browser_client = isinstance(client, BrowserClient)
            event_payload = client.get_event()
            team_names = team_names_from_event(event_payload)
            lineups = client.get_lineups()

        players = players_from_lineups(lineups, team_names)
        if not players:
            raise RuntimeError("Nenhum jogador encontrado no lineup.")

        lineup_rows = [
            {
                "match_id": match_id,
                "player_id": p.player_id,
                "player_name": p.player_name,
                "short_name": p.short_name,
                "position": p.position,
                "team": p.team,
                "side": p.side,
                "shirt_number": p.shirt_number,
                "substitute": p.substitute,
            }
            for p in players
        ]
        pd.DataFrame(lineup_rows).to_csv(output_dir / "lineups.csv", index=False)

        if client is None:
            client = build_client(mode, match_url, match_id, delay=delay)
            browser_client = isinstance(client, BrowserClient)

        all_rows: list[dict[str, Any]] = []
        selected_categories = tuple(categories)

        for index, player in enumerate(players, start=1):
            print(f"[{index}/{len(players)}] {player.player_name} ({player.player_id})")
            try:
                breakdown = client.get_rating_breakdown(player.player_id)
            except Exception as exc:  # noqa: BLE001
                print(f"  aviso: sem dados para {player.player_name} -> {exc}")
                continue

            if save_raw:
                raw_path = raw_dir / f"{player.player_id}_{_slug(player.player_name)}.json"
                raw_path.write_text(json.dumps(breakdown, ensure_ascii=False, indent=2), encoding="utf-8")

            for category in selected_categories:
                events = breakdown.get(category, [])
                if not isinstance(events, list):
                    continue
                all_rows.extend(flatten_category_events(events, category, player, match_id))

        if not all_rows:
            raise RuntimeError(
                "Nenhum evento extraído. A API pode estar bloqueada (403). "
                "Tente --mode browser ou exporte lineups.json manualmente e use --lineups."
            )

        all_df = pd.DataFrame(all_rows)
        passes_df = all_df[all_df["category"] == "passes"].copy()
        carries_df = all_df[all_df["category"] == "ball-carries"].copy()

        passes_df.to_csv(output_dir / "passes.csv", index=False)
        carries_df.to_csv(output_dir / "ball_carries.csv", index=False)
        all_df.to_csv(output_dir / "all_events.csv", index=False)

        summary = {
            "lineups": pd.DataFrame(lineup_rows),
            "passes": passes_df,
            "ball_carries": carries_df,
            "all_events": all_df,
        }
        print(
            f"\nConcluído: {len(passes_df)} passes, {len(carries_df)} conduções, "
            f"{len(players)} jogadores -> {output_dir}"
        )
        return summary
    finally:
        if browser_client and client is not None and hasattr(client, "close"):
            client.close()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "player"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extrai coordenadas e metadados de passes e conduções de uma partida SofaScore."
    )
    parser.add_argument(
        "match",
        nargs="?",
        help="URL da partida (com #id:...) ou match id numérico",
    )
    parser.add_argument(
        "--match-id",
        type=int,
        help="ID da partida (alternativa à URL)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("sofascore_output"),
        help="Pasta de saída (default: sofascore_output)",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "curl", "browser"),
        default="auto",
        help="Transporte HTTP: curl_cffi, browser (Chrome) ou auto",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.8,
        help="Pausa entre requisições, em segundos",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Salvar JSON bruto de rating-breakdown por jogador em output/raw/",
    )
    parser.add_argument(
        "--lineups",
        type=Path,
        help="JSON de lineups salvo manualmente (pula a chamada /lineups)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(EVENT_CATEGORIES),
        choices=EVENT_CATEGORIES,
        help="Categorias a extrair (default: passes ball-carries)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    match_ref = args.match
    if args.match_id is not None:
        match_ref = str(args.match_id)

    if not match_ref:
        if sys.stdin.isatty():
            print("Nenhuma partida informada.")
            print(f"Exemplo de URL:\n  {DEFAULT_MATCH_URL}\n")
            try:
                match_ref = input("Cole a URL da partida (ou match id) e pressione Enter: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelado.")
                return 1

    if not match_ref:
        parser.print_help()
        print(
            "\nErro: informe a URL da partida ou --match-id.\n"
            "\nNo terminal:\n"
            f'  python extract_sofascore_match_events.py "{DEFAULT_MATCH_URL}" -o argentina_austria\n'
            "\nNo VS Code: use o Terminal integrado com o comando acima, "
            "ou execute a configuração de debug \"SofaScore: extrair partida\"."
        )
        return 2

    try:
        extract_match_events(
            match_ref,
            args.output_dir,
            mode=args.mode,
            delay=args.delay,
            save_raw=args.save_raw,
            lineups_file=args.lineups,
            categories=args.categories,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
