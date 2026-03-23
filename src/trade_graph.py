"""
src/trade_graph.py
====================================================
Objective 2 — Graph representations of bilateral trade network structure.

Builds a weighted directed graph from trade_partners.csv where:
  - Nodes  = Zimbabwe + trading partner countries
  - Edges  = bilateral trade flows (imports: partner→Zimbabwe,
              exports: Zimbabwe→partner)
  - Weight = trade value (USD million)

Computes graph-derived node features for Zimbabwe per month:
  - graph_pagerank_zw       : PageRank score (network importance)
  - graph_n_active_partners : number of active trade partners
  - graph_network_density   : graph density (connectivity level)
  - graph_top_partner_share : largest single partner's share
  - graph_sadc_share        : SADC region's share of total flow

These are 'graph-derived node features' — a lightweight graph
representation tractable on 132 monthly observations. A full GNN
(e.g. GraphSAGE) is documented as future work: insufficient graph
snapshots exist for convolution to generalise at n=132.

Usage:
    from src.trade_graph import build_graph_features
    gdf = build_graph_features("data/raw/trade_partners.csv")
    # Returns DataFrame with columns: date, trade_type, graph_*
"""

import pandas as pd
import numpy as np
import networkx as nx
import os


SADC_COUNTRIES = {
    "South Africa", "Zambia", "Mozambique", "Botswana",
    "Zimbabwe", "Tanzania", "Malawi", "Namibia", "Angola",
    "Lesotho", "Eswatini", "Madagascar", "Mauritius",
    "Seychelles", "DRC", "Congo, DRC",
}


def _build_monthly_graph(partners_df: pd.DataFrame,
                          trade_type: str,
                          date: str) -> nx.DiGraph:
    """
    Build a weighted directed graph for one month and trade direction.
    Import:  partner → Zimbabwe
    Export:  Zimbabwe → partner
    """
    sub = partners_df[
        (partners_df["trade_type"] == trade_type) &
        (partners_df["date"] == date)
    ]
    G = nx.DiGraph()
    G.add_node("Zimbabwe", region="Zimbabwe")

    for _, row in sub.iterrows():
        partner = row["partner_country"]
        weight  = float(row["trade_value_mn_usd"])
        region  = row.get("region", "Other")
        G.add_node(partner, region=region)

        if trade_type == "import":
            G.add_edge(partner, "Zimbabwe", weight=weight)
        else:
            G.add_edge("Zimbabwe", partner, weight=weight)

    return G


def _extract_features(G: nx.DiGraph, trade_type: str) -> dict:
    """Extract Zimbabwe-centric graph features from a monthly graph."""
    total = sum(d["weight"] for _, _, d in G.edges(data=True)) + 1e-8

    # ── PageRank: Zimbabwe's importance in the network ──────────────────────
    try:
        pr = nx.pagerank(G, weight="weight", max_iter=200)
        zw_pagerank = pr.get("Zimbabwe", 0.0)
    except Exception:
        zw_pagerank = 0.0

    # ── Number of active partners (excluding aggregated 'Other') ─────────────
    active_partners = [
        n for n in G.nodes()
        if n not in ("Zimbabwe", "Other")
    ]
    n_active = len(active_partners)

    # ── Network density (how interconnected the graph is) ────────────────────
    density = nx.density(G)

    # ── Top partner share ────────────────────────────────────────────────────
    if trade_type == "import":
        zw_edges = [(u, d["weight"]) for u, v, d in G.in_edges("Zimbabwe", data=True)]
    else:
        zw_edges = [(v, d["weight"]) for u, v, d in G.out_edges("Zimbabwe", data=True)]

    top_share = 0.0
    if zw_edges:
        zw_edges_sorted = sorted(zw_edges, key=lambda x: -x[1])
        top_share = zw_edges_sorted[0][1] / total

    # ── SADC regional share ──────────────────────────────────────────────────
    sadc_weight = 0.0
    for node in G.nodes():
        region = G.nodes[node].get("region", "")
        if node in SADC_COUNTRIES or region == "SADC":
            if trade_type == "import":
                for u, v, d in G.in_edges("Zimbabwe", data=True):
                    if u == node:
                        sadc_weight += d["weight"]
            else:
                for u, v, d in G.out_edges("Zimbabwe", data=True):
                    if v == node:
                        sadc_weight += d["weight"]

    sadc_share = sadc_weight / total

    return {
        "graph_pagerank_zw":       round(zw_pagerank, 6),
        "graph_n_active_partners": n_active,
        "graph_network_density":   round(density, 6),
        "graph_top_partner_share": round(top_share, 6),
        "graph_sadc_share":        round(sadc_share, 6),
    }


def build_graph_features(partners_path: str) -> pd.DataFrame:
    """
    Build monthly graph features for both import and export directions.

    Parameters
    ----------
    partners_path : str
        Path to trade_partners.csv

    Returns
    -------
    pd.DataFrame
        Columns: date, trade_type, graph_pagerank_zw,
                 graph_n_active_partners, graph_network_density,
                 graph_top_partner_share, graph_sadc_share
    """
    partners = pd.read_csv(partners_path)
    dates = sorted(partners["date"].unique())
    rows  = []

    for date in dates:
        for trade_type in ["import", "export"]:
            G    = _build_monthly_graph(partners, trade_type, date)
            feat = _extract_features(G, trade_type)
            feat["date"]       = date
            feat["trade_type"] = trade_type
            rows.append(feat)

    df = pd.DataFrame(rows)
    print(f"  Graph features built: {len(df)} rows × {len(df.columns)} columns")
    print(f"  Features: {[c for c in df.columns if c.startswith('graph_')]}")
    return df


def save_graph_features(partners_path: str,
                         output_path: str = "outputs/tables/graph_features.csv") -> pd.DataFrame:
    """Build and save graph features to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = build_graph_features(partners_path)
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    return df


def plot_trade_graph(partners_path: str,
                     trade_type: str = "export",
                     date: str = "2022-01",
                     output_path: str = "outputs/plots/trade_network_graph.png"):
    """
    Visualise the bilateral trade network for one month.
    Node size = trade value. Zimbabwe is highlighted.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    partners = pd.read_csv(partners_path)
    G = _build_monthly_graph(partners, trade_type, date)

    # Layout: Zimbabwe at centre
    pos = nx.spring_layout(G, seed=42, k=2.5)
    pos["Zimbabwe"] = np.array([0.0, 0.0])

    total = sum(d["weight"] for _, _, d in G.edges(data=True)) + 1e-8

    # Node sizes: Zimbabwe large, others proportional to flow
    node_sizes = []
    node_colors = []
    for node in G.nodes():
        if node == "Zimbabwe":
            node_sizes.append(3000)
            node_colors.append("#1d9e75")
        else:
            region = G.nodes[node].get("region", "Other")
            if trade_type == "import":
                weight = sum(d["weight"] for u, v, d in G.in_edges("Zimbabwe", data=True) if u == node)
            else:
                weight = sum(d["weight"] for u, v, d in G.out_edges("Zimbabwe", data=True) if v == node)
            node_sizes.append(max(300, weight / total * 5000))
            region_colors = {
                "SADC": "#ef9f27", "Asia": "#7f77dd",
                "Europe": "#5dcaa5", "Americas": "#d85a30",
                "Middle East": "#afa9ec", "Africa": "#ef9f27",
            }
            node_colors.append(region_colors.get(region, "#888780"))

    # Edge weights → line width
    edge_weights = [G[u][v]["weight"] / total * 15 for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=node_colors, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_weights,
                           edge_color="#5dcaa5", alpha=0.6,
                           arrows=True, arrowsize=20,
                           connectionstyle="arc3,rad=0.1", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9,
                            font_color="#e8e6e0", ax=ax)

    legend_patches = [
        mpatches.Patch(color="#1d9e75", label="Zimbabwe (focal node)"),
        mpatches.Patch(color="#ef9f27", label="SADC / Africa"),
        mpatches.Patch(color="#7f77dd", label="Asia"),
        mpatches.Patch(color="#5dcaa5", label="Europe"),
        mpatches.Patch(color="#d85a30", label="Americas"),
        mpatches.Patch(color="#afa9ec", label="Middle East"),
    ]
    ax.legend(handles=legend_patches, fontsize=8,
              framealpha=0.2, loc="lower left",
              labelcolor="#e8e6e0")

    direction = "imports from" if trade_type == "import" else "exports to"
    ax.set_title(f"Zimbabwe bilateral trade network — {direction} partners ({date})",
                 color="#e8e6e0", fontsize=12, fontweight="bold", pad=12)
    ax.axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    print("Building trade graph features...")
    gdf = save_graph_features("data/raw/trade_partners.csv")
    print(gdf.head(6).to_string())

    print("\nPlotting import network (2022-01)...")
    plot_trade_graph("data/raw/trade_partners.csv",
                     trade_type="import", date="2022-01",
                     output_path="outputs/plots/trade_network_import.png")
    plot_trade_graph("data/raw/trade_partners.csv",
                     trade_type="export", date="2022-01",
                     output_path="outputs/plots/trade_network_export.png")
    print("Done.")
