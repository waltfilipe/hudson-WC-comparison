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
API_BASE = "https://api.sofascore.com/api/v1"
API_BASES = (
    "https://www.sofascore.com/api/v1",
    "https://api.sofascore.com/api/v1",
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


def _first_present(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _extract_json_text(raw_text: str) -> str:
    text = raw_text.strip().lstrip("\ufeff")
    if not text:
        raise ValueError("arquivo vazio")

    if text.startswith("{") or text.startswith("["):
        return text

    if "<pre" in text.lower():
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(text, "html.parser")
        pre = soup.find("pre")
        if pre and pre.get_text(strip=True):
            return pre.get_text()

    match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if match:
        return match.group(1)

    raise ValueError("não encontrei JSON válido no arquivo")


def _normalize_lineups_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("o JSON precisa ser um objeto com as chaves 'home' e 'away'")

    if "home" in payload and "away" in payload:
        return payload

    for key in ("lineups", "data", "response"):
        nested = payload.get(key)
        if isinstance(nested, dict) and "home" in nested and "away" in nested:
            return nested

    raise ValueError(
        "formato inválido: esperado {'home': {'players': [...]}, 'away': {'players': [...]}}"
    )


def load_lineups_file(lineups_file: Path, base_dir: Path | None = None) -> dict[str, Any]:
    path = Path(lineups_file)
    if not path.is_absolute() and base_dir is not None:
        candidate = base_dir / path
        if candidate.exists():
            path = candidate

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            "Salve o JSON na mesma pasta do notebook/script ou informe o caminho completo."
        )

    raw_text = path.read_text(encoding="utf-8")
    if not raw_text.strip():
        raise ValueError(
            f"O arquivo está vazio: {path}\n"
            "Exporte novamente pelo DevTools (veja instruções no notebook)."
        )

    try:
        json_text = _extract_json_text(raw_text)
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        preview = raw_text[:120].replace("\n", " ")
        raise ValueError(
            f"JSON inválido em {path}: {exc}\n"
            f"Início do arquivo: {preview!r}\n"
            "Você provavelmente salvou HTML ou copiou texto errado. "
            "Use DevTools → Network → lineups → Response → Copy response."
        ) from exc

    lineups = _normalize_lineups_payload(payload)
    if not players_from_lineups(lineups):
        raise ValueError(
            f"O arquivo {path} foi lido, mas não contém jogadores em home/away.players."
        )
    return lineups


def _is_access_denied(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "403" in text or "forbidden" in text or "challenge" in text


def _chrome_major_version() -> int | None:
    import re
    import subprocess

    if sys.platform == "win32":
        commands = [
            ["reg", "query", r"HKEY_CURRENT_USER\Software\Google\Chrome\BLBeacon", "/v", "version"],
            ["reg", "query", r"HKEY_LOCAL_MACHINE\SOFTWARE\Google\Chrome\BLBeacon", "/v", "version"],
        ]
    elif sys.platform == "darwin":
        commands = [["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--version"]]
    else:
        commands = [
            ["google-chrome", "--version"],
            ["google-chrome-stable", "--version"],
            ["chromium", "--version"],
            ["chromium-browser", "--version"],
        ]

    for command in commands:
        try:
            output = subprocess.run(command, capture_output=True, text=True, timeout=5).stdout
            match = re.search(r"(\d+)\.", output)
            if match:
                return int(match.group(1))
        except Exception:
            continue
    return None


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

    def __init__(
        self,
        match_url: str,
        match_id: int,
        delay: float = 1.5,
        *,
        headless: bool = True,
    ) -> None:
        import undetected_chromedriver as uc

        self.match_id = match_id
        self.match_url = match_url
        self.delay = delay
        self.headless = headless

        options = uc.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--window-size=1440,900")

        chrome_version = _chrome_major_version()
        self.driver = uc.Chrome(
            options=options,
            version_main=chrome_version,
        )
        self.driver.get(self.match_url)
        time.sleep(self.delay)

    def close(self) -> None:
        self.driver.quit()

    def get_json(self, path: str) -> dict[str, Any]:
        rel_path = path.lstrip("/")
        last_error: Exception | None = None

        for base in API_BASES:
            url = f"{base}/{rel_path}"
            try:
                payload = self._fetch_via_browser(url)
                if isinstance(payload, dict) and payload.get("error"):
                    error = payload["error"]
                    if error.get("code") == 403:
                        last_error = RuntimeError(f"403 Forbidden em {url}: {error}")
                        continue
                    raise RuntimeError(f"Erro da API em {url}: {error}")
                time.sleep(self.delay)
                return payload
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        raise RuntimeError(f"Falha ao acessar {rel_path} via browser") from last_error

    def _fetch_via_browser(self, url: str) -> dict[str, Any]:
        script = """
        const url = arguments[0];
        const done = arguments[arguments.length - 1];
        fetch(url, {
            method: 'GET',
            credentials: 'include',
            headers: {
                'Accept': 'application/json, text/plain, */*',
            },
        })
        .then(async (response) => {
            let data = null;
            try {
                data = await response.json();
            } catch (error) {
                data = {error: {message: 'invalid json', status: response.status}};
            }
            done({ok: response.ok, status: response.status, data});
        })
        .catch((error) => done({ok: false, status: 0, data: {error: {message: String(error)}}}));
        """
        result = self.driver.execute_async_script(script, url)
        if not isinstance(result, dict):
            raise RuntimeError(f"Resposta inesperada do browser para {url}")

        status = result.get("status", 0)
        payload = result.get("data", {})
        if status == 403 or (isinstance(payload, dict) and payload.get("error", {}).get("code") == 403):
            raise RuntimeError(f"403 Forbidden em {url}")
        if not result.get("ok"):
            raise RuntimeError(f"HTTP {status} em {url}: {payload}")
        if not isinstance(payload, dict):
            raise RuntimeError(f"JSON inválido em {url}")
        return payload

    def get_lineups(self) -> dict[str, Any]:
        return self.get_json(f"event/{self.match_id}/lineups")

    def get_event(self) -> dict[str, Any]:
        return self.get_json(f"event/{self.match_id}")

    def get_rating_breakdown(self, player_id: int) -> dict[str, Any]:
        return self.get_json(f"event/{self.match_id}/player/{player_id}/rating-breakdown")


def create_working_client(
    mode: str,
    match_url: str,
    match_id: int,
    delay: float,
) -> tuple[Any, str]:
    """Tenta curl e/ou Chrome até conseguir acessar a API."""
    attempts: list[tuple[str, dict[str, Any]]] = []

    if mode == "curl":
        attempts.append(("curl", {}))
    elif mode == "browser":
        attempts.append(("browser-headless", {"headless": True}))
        attempts.append(("browser-visible", {"headless": False}))
    else:
        attempts.extend(
            [
                ("curl", {}),
                ("browser-headless", {"headless": True}),
                ("browser-visible", {"headless": False}),
            ]
        )

    last_error: Exception | None = None
    for label, options in attempts:
        client = None
        try:
            if label == "curl":
                client = CurlCffiClient(match_url, match_id, delay=delay)
            else:
                client = BrowserClient(match_url, match_id, delay=delay, **options)
            client.get_event()
            print(f"Conectado via {label}")
            return client, label
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"Modo {label} falhou: {exc}")
            if client is not None and hasattr(client, "close"):
                client.close()
            if mode in {"curl", "browser"}:
                break
            if mode == "auto" and label == "curl" and not _is_access_denied(exc):
                break

    raise RuntimeError(
        "Não foi possível acessar a API do SofaScore (403).\n"
        "Tente:\n"
        "  1) MODE = 'browser' no notebook (abre o Chrome)\n"
        "  2) Instalar/atualizar o Google Chrome\n"
        "  3) Exportar lineups.json no DevTools e usar --lineups"
    ) from last_error


def build_client(mode: str, match_url: str, match_id: int, delay: float):
    client, _ = create_working_client(mode, match_url, match_id, delay)
    return client


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
    client_label = mode
    browser_client = False

    try:
        if lineups_file:
            lineups = load_lineups_file(lineups_file, base_dir=output_dir.parent)
            team_names = {}
        else:
            client, client_label = create_working_client(mode, match_url, match_id, delay=delay)
            browser_client = client_label.startswith("browser")
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
            client, client_label = create_working_client(mode, match_url, match_id, delay=delay)
            browser_client = client_label.startswith("browser")

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
