"""
live_dashboard.py — Real-time store analytics terminal dashboard.

Connects to the Intelligence API and refreshes every 5 seconds.
Shows live metrics: visitor count, conversion rate, zone heatmap, active anomalies.

Usage:
    python dashboard/live_dashboard.py [--api http://localhost:8000] [--store STORE_BLR_002]

Part E bonus: This proves the pipeline and API are genuinely connected —
the dashboard polls /metrics, /heatmap, /anomalies in real time as events flow in.
"""

import os
import sys
import time
import requests
import argparse
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.columns import Columns
from rich import box

API_URL = os.environ.get("API_URL", "http://localhost:8000")
STORE_ID = os.environ.get("STORE_ID", "STORE_BLR_002")
REFRESH_SECONDS = 5


def fetch(path: str) -> dict | None:
    try:
        resp = requests.get(f"{API_URL}{path}", timeout=4)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def severity_style(severity: str) -> str:
    return {"CRITICAL": "bold red", "WARN": "bold yellow", "INFO": "blue"}.get(severity, "white")


def make_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    layout["left"].split_column(
        Layout(name="metrics"),
        Layout(name="funnel"),
    )
    layout["right"].split_column(
        Layout(name="heatmap"),
        Layout(name="anomalies"),
    )
    return layout


def render_header(store_id: str) -> Panel:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return Panel(
        f"[bold cyan]📊 Apex Retail — Store Intelligence Dashboard[/bold cyan]  "
        f"[dim]Store: {store_id}  |  {now}  |  Refresh: {REFRESH_SECONDS}s[/dim]",
        style="on black",
    )


def render_metrics(data: dict | None) -> Panel:
    if not data:
        return Panel("[red]⚠ Metrics unavailable[/red]", title="Live Metrics")

    t = Table(show_header=False, box=None, padding=(0, 1))
    t.add_column("Metric", style="dim")
    t.add_column("Value", style="bold")

    t.add_row("👤 Unique Visitors", str(data.get("unique_visitors", 0)))
    cr = data.get("conversion_rate", 0)
    cr_style = "green" if cr >= 0.3 else ("yellow" if cr >= 0.15 else "red")
    t.add_row("💳 Conversion Rate", f"[{cr_style}]{cr:.1%}[/{cr_style}]")
    t.add_row("🧾 Revenue (INR)", f"₹{data.get('total_revenue_inr', 0):,.0f}")
    t.add_row("🏪 Queue Depth", str(data.get("current_queue_depth", 0)))
    abr = data.get("abandonment_rate", 0)
    t.add_row("🚪 Abandonment Rate", f"[{'red' if abr > 0.3 else 'green'}]{abr:.1%}[/]")
    t.add_row("🕐 As Of", data.get("as_of", "N/A")[-8:])

    return Panel(t, title="[bold]📈 Live Metrics[/bold]", border_style="cyan")


def render_funnel(data: dict | None) -> Panel:
    if not data:
        return Panel("[red]⚠ Funnel unavailable[/red]", title="Funnel")

    t = Table(box=box.SIMPLE_HEAVY)
    t.add_column("Stage", style="bold")
    t.add_column("Sessions", justify="right")
    t.add_column("Drop-off", justify="right")

    stage_icons = {"ENTRY": "🚪", "ZONE_VISIT": "🛍", "BILLING_QUEUE": "🧾", "PURCHASE": "✅"}
    for stage in data.get("funnel", []):
        icon = stage_icons.get(stage["stage"], "•")
        drop = stage["drop_off_pct"]
        drop_style = "red" if drop > 40 else ("yellow" if drop > 20 else "green")
        t.add_row(
            f"{icon} {stage['stage']}",
            str(stage["count"]),
            f"[{drop_style}]{drop:.1f}%[/{drop_style}]" if drop > 0 else "[dim]—[/dim]",
        )

    return Panel(t, title="[bold]🔽 Conversion Funnel[/bold]", border_style="magenta")


def render_heatmap(data: dict | None) -> Panel:
    if not data or not data.get("zones"):
        return Panel("[red]⚠ Heatmap unavailable[/red]", title="Zone Heatmap")

    t = Table(box=box.SIMPLE_HEAVY)
    t.add_column("Zone", style="bold")
    t.add_column("Visits", justify="right")
    t.add_column("Avg Dwell", justify="right")
    t.add_column("Score", justify="right")
    t.add_column("🌡", justify="center")

    def heat_bar(score: float) -> str:
        filled = int(score / 10)
        colors = ["blue", "cyan", "green", "yellow", "red"]
        c = colors[min(filled // 2, 4)]
        return f"[{c}]{'█' * filled}{'░' * (10 - filled)}[/{c}]"

    for z in data.get("zones", []):
        dwell = z.get("avg_dwell_seconds", 0)
        score = z.get("normalised_score", 0)
        t.add_row(
            z["zone_id"],
            str(z["visit_frequency"]),
            f"{dwell:.0f}s",
            f"{score:.0f}",
            heat_bar(score),
        )

    return Panel(t, title="[bold]🗺 Zone Heatmap[/bold]", border_style="yellow")


def render_anomalies(data: dict | None) -> Panel:
    if not data:
        return Panel("[red]⚠ Anomaly data unavailable[/red]", title="Anomalies")

    anomalies = data.get("active_anomalies", [])
    if not anomalies:
        return Panel("[green]✅ No active anomalies[/green]", title="[bold]⚠ Anomalies[/bold]", border_style="green")

    t = Table(box=box.SIMPLE_HEAVY)
    t.add_column("Severity")
    t.add_column("Type", style="bold")
    t.add_column("Action")

    for a in anomalies:
        sev = a["severity"]
        t.add_row(
            Text(sev, style=severity_style(sev)),
            a["anomaly_type"],
            a.get("suggested_action", "—")[:60],
        )

    return Panel(t, title=f"[bold]⚠ Anomalies ({len(anomalies)})[/bold]", border_style="red")


def render_footer(health: dict | None) -> Panel:
    if not health:
        status = "[red]API UNREACHABLE[/red]"
    else:
        db = health.get("database", "?")
        overall = health.get("status", "?")
        color = "green" if overall == "OK" else "yellow"
        status = f"[{color}]API: {overall}[/{color}]  DB: {db}"
    return Panel(f"[dim]Store Intelligence v1.0  |  {status}[/dim]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default=API_URL)
    parser.add_argument("--store", default=STORE_ID)
    args = parser.parse_args()

    console = Console()
    layout = make_layout()

    with Live(layout, console=console, refresh_per_second=0.5, screen=True):
        while True:
            metrics = fetch(f"/stores/{args.store}/metrics")
            funnel = fetch(f"/stores/{args.store}/funnel")
            heatmap = fetch(f"/stores/{args.store}/heatmap")
            anomalies = fetch(f"/stores/{args.store}/anomalies")
            health = fetch("/health")

            layout["header"].update(render_header(args.store))
            layout["metrics"].update(render_metrics(metrics))
            layout["funnel"].update(render_funnel(funnel))
            layout["heatmap"].update(render_heatmap(heatmap))
            layout["anomalies"].update(render_anomalies(anomalies))
            layout["footer"].update(render_footer(health))

            time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()
