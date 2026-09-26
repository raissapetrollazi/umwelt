"""Offline, read-only HTML dashboard for the frozen v0.4 social result."""

from __future__ import annotations

import json
from html import escape
from math import isfinite
from pathlib import Path
from typing import cast

from umwelt.errors import DataError
from umwelt.social_model import SOCIAL_MODEL_IDS
from umwelt.spatial_reporting import file_sha256


def _verified_directory(directory: Path) -> dict[str, object]:
    manifest_path = directory / "artifact-manifest.json"
    if not manifest_path.is_file():
        raise DataError(f"Social artifact manifest is missing: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("artifacts") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        raise DataError("Social artifact manifest is invalid.")
    for entry in entries:
        if not isinstance(entry, dict):
            raise DataError("Social artifact entry is invalid.")
        name, expected = entry.get("name"), entry.get("sha256")
        if not isinstance(name, str) or not isinstance(expected, str):
            raise DataError("Social artifact entry is invalid.")
        if Path(name).name != name:
            raise DataError("Social artifact name is invalid.")
        path = directory / name
        if not path.is_file() or file_sha256(path) != expected:
            raise DataError(f"Social artifact hash differs: {name}")
    required = {"comparison.json", "provenance.json", "recorded-metrics.jsonl"}
    if not required.issubset({entry["name"] for entry in entries}):
        raise DataError("Social dashboard needs complete result artifacts.")
    return cast(dict[str, object], manifest)


def _json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DataError(f"Social result JSON must be an object: {path.name}")
    return cast(dict[str, object], value)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataError("Social dashboard metric must be numeric.")
    number = float(value)
    if not isfinite(number) or number < 0:
        raise DataError("Social dashboard metric must be finite and non-negative.")
    return number


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{_number(value):,.{digits}f}".replace(",", " ")


def _pair_name(pair_id: object) -> str:
    if not isinstance(pair_id, str):
        raise DataError("Social dashboard pair identifier is invalid.")
    return escape(pair_id.rsplit("/", 1)[-1].split("_", 1)[0])


def _phase_html(name: str, directory: Path) -> str:
    comparison = _json_object(directory / "comparison.json")
    means = comparison.get("equal_pair_mean_distance_wasserstein_px")
    changes = comparison.get("equal_pair_mean_signed_change_wasserstein_px")
    per_pair = comparison.get("per_pair")
    if (
        not isinstance(means, dict)
        or not isinstance(changes, dict)
        or not isinstance(per_pair, list)
    ):
        raise DataError("Social comparison lacks dashboard fields.")
    baseline, candidate = SOCIAL_MODEL_IDS
    baseline_distance = _number(means[baseline])
    candidate_distance = _number(means[candidate])
    baseline_change = _number(changes[baseline])
    candidate_change = _number(changes[candidate])
    wins = comparison.get("candidate_distance_win_count")
    eligible = comparison.get("eligible_pair_count")
    required = comparison.get("required_distance_win_count")
    if not all(type(value) is int for value in (wins, eligible, required)):
        raise DataError("Social comparison pair exposure is invalid.")
    preferred = comparison.get("candidate_preferred")
    if not isinstance(preferred, bool):
        raise DataError("Social dashboard requires a completed preference decision.")
    saved_fraction = (
        100 * (baseline_distance - candidate_distance) / baseline_distance
        if baseline_distance > 0
        else 0.0
    )

    if not per_pair:
        raise DataError("Social dashboard needs at least one pair result.")
    pairs: list[tuple[str, float, float]] = []
    for pair in per_pair:
        if not isinstance(pair, dict):
            raise DataError("Social per-pair result is invalid.")
        models = pair.get("models")
        if not isinstance(models, dict) or not all(
            isinstance(models.get(model_id), dict)
            and "distance_wasserstein_px" in models[model_id]
            for model_id in SOCIAL_MODEL_IDS
        ):
            raise DataError("Social per-pair model results are invalid.")
        s0 = _number(
            cast(dict[str, object], models[baseline])["distance_wasserstein_px"]
        )
        s1 = _number(
            cast(dict[str, object], models[candidate])["distance_wasserstein_px"]
        )
        pair_label = _pair_name(pair.get("pair_id"))
        pairs.append((pair_label, s0, s1))
    maximum = max((value for _, s0, s1 in pairs for value in (s0, s1)), default=1.0)
    maximum = maximum or 1.0
    bars = []
    for pair_label, s0, s1 in pairs:
        note = "S1 melhor" if s1 < s0 else "S0 melhor ou empate"
        bars.append(
            f'<div class="pair-row"><div class="pair-label">{pair_label}</div>'
            f'<div class="pair-bars"><div class="bar-line"><span>S0</span><div class="track"><div class="bar bar-s0" style="width:{100 * s0 / maximum:.3f}%"></div></div><strong>{_fmt(s0)}</strong></div>'
            f'<div class="bar-line"><span>S1</span><div class="track"><div class="bar bar-s1" style="width:{100 * s1 / maximum:.3f}%"></div></div><strong>{_fmt(s1)}</strong></div></div>'
            f'<div class="pair-note">{note}</div></div>'
        )
    phase_id = "development" if name == "Desenvolvimento" else "test"
    verdict = "Critério atendido" if preferred else "Critério não atendido"
    distance_change = (
        f"redução de {saved_fraction:.1f}%"
        if saved_fraction >= 0
        else f"aumento de {-saved_fraction:.1f}%"
    )
    return f"""
<section id="{phase_id}" class="phase-section" aria-labelledby="{phase_id}-title">
  <div class="section-head"><div><p class="eyebrow">{name}</p><h2 id="{phase_id}-title">{verdict}</h2></div><p class="section-note">Unidade: par gravado. Oito réplicas por modelo e por par.</p></div>
  <div class="metric-grid">
    <article class="metric-card"><span>Erro na distância entre animais</span><strong>{_fmt(baseline_distance)} <small>→</small> {_fmt(candidate_distance)} <em>px</em></strong><p>W1 médio por par, S0 → S1 · {distance_change}</p></article>
    <article class="metric-card"><span>Erro na mudança de distância</span><strong>{_fmt(baseline_change, 3)} <small>→</small> {_fmt(candidate_change, 3)} <em>px</em></strong><p>W1 médio por par, S0 → S1</p></article>
    <article class="metric-card"><span>Pares com menor erro em S1</span><strong>{wins} <small>/</small> {eligible}</strong><p>O critério exigia pelo menos {required} pares.</p></article>
  </div>
  <div class="chart-head"><h3>Erro de distância por par</h3><p>Menor barra indica distribuição sintética mais próxima da observada. Valores em pixels.</p></div>
  <div class="pair-chart">{"".join(bars)}</div>
</section>"""


def render_social_dashboard(
    development_directory: str | Path,
    test_directory: str | Path,
    *,
    output_path: str | Path,
) -> Path:
    """Write one self-contained HTML summary without raw pose coordinates."""

    development = Path(development_directory).resolve()
    test = Path(test_directory).resolve()
    target = Path(output_path).resolve()
    if target.exists():
        raise DataError(f"Social dashboard output already exists: {target}")
    _verified_directory(development)
    _verified_directory(test)
    development_provenance = _json_object(development / "provenance.json")
    test_provenance = _json_object(test / "provenance.json")
    if development_provenance.get("source") != test_provenance.get("source") or (
        development_provenance.get("frozen_protocol_document_sha256")
        != test_provenance.get("frozen_protocol_document_sha256")
    ):
        raise DataError("Social dashboard phases use different source or protocol.")
    if test_provenance.get("development_manifest_sha256") != file_sha256(
        development / "artifact-manifest.json"
    ):
        raise DataError(
            "Social dashboard test does not reference these development artifacts."
        )
    development_software = development_provenance.get("software")
    test_software = test_provenance.get("software")
    if (
        not isinstance(development_software, dict)
        or not isinstance(test_software, dict)
        or development_software.get("git_revision") != test_software.get("git_revision")
    ):
        raise DataError("Social dashboard phases use different software revisions.")
    development_comparison = _json_object(development / "comparison.json")
    test_comparison = _json_object(test / "comparison.json")
    decisions = (
        development_comparison.get("candidate_preferred"),
        test_comparison.get("candidate_preferred"),
    )
    if not all(isinstance(value, bool) for value in decisions):
        raise DataError("Social dashboard needs completed split decisions.")
    if decisions == (True, True):
        verdict = "S1 atendeu ao critério nos dois conjuntos."
    elif decisions == (False, False):
        verdict = "S1 não atendeu ao critério em nenhum dos conjuntos."
    else:
        verdict = "As decisões de desenvolvimento e teste diferiram."
    revision = cast(dict[str, object], test_provenance.get("software", {})).get(
        "git_revision"
    )
    if not isinstance(revision, str):
        raise DataError("Social dashboard lacks a software revision.")
    development_html = _phase_html("Desenvolvimento", development)
    test_html = _phase_html("Teste retido", test)
    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Umwelt · laboratório social v0.4</title>
<style>
:root {{ color-scheme: light; --ink:#17302d; --muted:#526562; --paper:#f7f5ef; --card:#fffdfa; --line:#d7ded6; --s0:#9caeaa; --s1:#1f806c; --accent:#d0e7dd; }}
* {{ box-sizing:border-box; }} html {{ scroll-behavior:smooth; }} body {{ margin:0; color:var(--ink); background:var(--paper); font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }}
header {{ background:#113d36; color:white; padding:28px max(24px,calc((100vw - 1100px)/2)); }}
.brand {{ letter-spacing:.18em; text-transform:uppercase; font-size:.78rem; font-weight:800; color:#b7e4d2; }}
h1 {{ font-size:clamp(2rem,5vw,3.6rem); line-height:1.1; margin:12px 0; letter-spacing:-.035em; }}
header p {{ max-width:70ch; color:#d5e6df; margin:0; }}
nav {{ display:flex; gap:12px; flex-wrap:wrap; margin-top:24px; }} nav a {{ color:white; text-decoration:none; border:1px solid #83b3a4; border-radius:999px; padding:8px 15px; font-weight:650; }} nav a:hover, nav a:focus-visible {{ background:#316d5c; outline:2px solid white; outline-offset:2px; }}
main {{ max-width:1100px; padding:32px 24px 70px; margin:auto; }}
.notice {{ border-left:5px solid var(--s1); background:#e6f0e8; padding:16px 20px; border-radius:0 12px 12px 0; margin-bottom:38px; }}
.notice strong {{ display:block; margin-bottom:3px; }}
.phase-section {{ scroll-margin-top:18px; margin-bottom:65px; }} .section-head {{ display:flex; justify-content:space-between; gap:24px; align-items:end; border-bottom:1px solid var(--line); margin-bottom:20px; }}
.eyebrow {{ text-transform:uppercase; letter-spacing:.15em; font-size:.75rem; font-weight:800; color:var(--s1); margin:0; }} h2 {{ font-size:clamp(1.7rem,3vw,2.4rem); margin:3px 0 12px; letter-spacing:-.025em; }} .section-note {{ color:var(--muted); max-width:33ch; text-align:right; }}
.metric-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }} .metric-card {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:20px; min-height:158px; box-shadow:0 4px 18px #17302d08; }}
.metric-card span {{ color:var(--muted); font-size:.86rem; font-weight:650; }} .metric-card strong {{ display:block; font-size:clamp(1.2rem,2vw,1.8rem); margin:10px 0 5px; line-height:1.2; white-space:nowrap; }} .metric-card small {{ color:#8b9b96; font-size:.75em; }} .metric-card em {{ font-size:.55em; font-style:normal; font-weight:500; }} .metric-card p {{ color:var(--muted); margin:0; font-size:.83rem; }}
.chart-head {{ display:flex; justify-content:space-between; align-items:baseline; gap:20px; margin:30px 0 8px; }} h3 {{ font-size:1.35rem; margin:0; }} .chart-head p {{ color:var(--muted); font-size:.9rem; margin:0; max-width:43ch; }}
.pair-chart {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:12px 20px; }} .pair-row {{ display:grid; grid-template-columns:95px minmax(0,1fr) 145px; gap:16px; align-items:center; padding:12px 0; border-bottom:1px solid #e5eae4; }} .pair-row:last-child {{ border-bottom:0; }} .pair-label {{ font-weight:750; }} .bar-line {{ display:grid; grid-template-columns:24px minmax(0,1fr) 68px; align-items:center; gap:8px; line-height:1.35; font-size:.77rem; }} .bar-line span {{ color:var(--muted); }} .bar-line strong {{ text-align:right; font-variant-numeric:tabular-nums; }} .track {{ height:9px; background:#eaf0ec; border-radius:99px; overflow:hidden; }} .bar {{ height:100%; border-radius:99px; min-width:1px; }} .bar-s0 {{ background:var(--s0); }} .bar-s1 {{ background:var(--s1); }} .pair-note {{ color:var(--muted); font-size:.79rem; text-align:right; }}
footer {{ max-width:1100px; margin:auto; padding:0 24px 50px; color:var(--muted); border-top:1px solid var(--line); }} footer h2 {{ color:var(--ink); font-size:1.5rem; margin-top:27px; }} footer code {{ overflow-wrap:anywhere; }} footer p {{ max-width:82ch; }} footer a {{ color:#116a59; }}
@media(max-width:760px) {{ .metric-grid {{ grid-template-columns:1fr; }} .section-head,.chart-head {{ display:block; }} .section-note {{ text-align:left; }} .pair-row {{ grid-template-columns:72px minmax(0,1fr); gap:9px; }} .pair-note {{ grid-column:2; text-align:left; }} .pair-chart {{ padding:10px 12px; }} .metric-card {{ min-height:auto; }} }}
</style>
</head>
<body>
<header><div class="brand">Umwelt / laboratório computacional</div><h1>Dois animais, uma pergunta social.</h1><p>Comparação de dois modelos gerativos de distância entre camundongos. S0 move os animais independentemente; S1 acrescenta uma resposta direcional do residente à posição atual do intruso.</p><nav aria-label="Seções"><a href="#development">Desenvolvimento</a><a href="#test">Teste retido</a><a href="#methods">Método e limites</a></nav></header>
<main><aside class="notice"><strong>{verdict}</strong>Compare as duas métricas e cada par. A preferência do critério não demonstra intenção ou mecanismo social biológico.</aside>{development_html}{test_html}</main>
<footer id="methods"><h2>Método e limites</h2><p>W1 é a distância de Wasserstein entre a distribuição gravada e a sintética; menor é melhor. O valor de cada modelo é a média de oito réplicas dentro de cada par, seguida da média entre pares. O critério foi fixado antes da avaliação: pelo menos 5% menos erro de distância, no máximo 5% mais erro de mudança de distância, e melhora em pelo menos 60% dos pares.</p><p>Os limites da imagem são restrições computacionais, não paredes medidas da gaiola. As identidades globais dos intrusos não estão disponíveis nos nomes do conjunto. Os dados CalMS21 têm licença CC-BY-NC-SA, conforme o <a href="https://arxiv.org/html/2104.02710v4">artigo do conjunto</a>; este painel contém apenas resumos derivados e deve permanecer no uso local de pesquisa.</p><p>Fonte: <a href="https://data.caltech.edu/records/s0vdx-0k302">CalMS21 Task 1</a> · Revisão do experimento: <code>{escape(revision)}</code>.</p></footer>
</body></html>
"""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return target
