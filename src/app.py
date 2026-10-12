"""Cricket Scout AI: interactive scouting dashboard."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Cricket Scout AI", page_icon="🏏", layout="wide")

ACCENT = "#22d3a6"
STYLE_COLORS = {
    "Aggressive powerplay hitter": "#ff6b6b",
    "Power finisher": "#f7b731",
    "Anchor": "#4dabf7",
    "Traditional opener": "#22d3a6",
    "Middle-order accelerator": "#b197fc",
}

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
.hero {padding: 1.6rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem;
       background: linear-gradient(135deg, #12304a 0%, #0f5e55 100%);}
.hero h1 {margin: 0; font-size: 2.4rem;}
.hero p {margin: .4rem 0 0 0; opacity: .85; font-size: 1.02rem;}
.card {background: #131c2e; border: 1px solid #22304a; border-radius: 14px;
       padding: 1rem 1.2rem; text-align: center;}
.card .num {font-size: 1.9rem; font-weight: 700; color: #22d3a6;}
.card .lab {font-size: .85rem; opacity: .7;}
.verdict {background: #131c2e; border-left: 4px solid #22d3a6; border-radius: 8px;
          padding: .9rem 1.1rem; margin: .6rem 0 1rem 0;}
</style>
""",
    unsafe_allow_html=True,
)


def style_fig(fig, height=480):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=dict(l=10, r=10, t=30, b=10), font=dict(size=13),
    )
    return fig


def card(col, num, label):
    col.markdown(f'<div class="card"><div class="num">{num}</div><div class="lab">{label}</div></div>',
                 unsafe_allow_html=True)


@st.cache_data
def load():
    rae = pd.read_parquet("data/app/batter_rae.parquet")
    cl = pd.read_parquet("data/app/batter_clusters.parquet")
    cl["boundary_pct"] = cl["four_pct"] + cl["six_pct"]
    return rae, cl


rae, cl = load()

AXES = {  # label -> (column, higher_is_better)
    "Scoring rate": ("rel_sr", True),
    "Boundary hitting": ("boundary_pct", True),
    "Six power": ("six_pct", True),
    "Dot-ball avoidance": ("dot_pct", False),
    "Staying power": ("balls_per_out", True),
    "Powerplay usage": ("pp_share", True),
    "Death-over usage": ("death_share", True),
}
pct = pd.DataFrame(index=cl.index)
for label, (col, hib) in AXES.items():
    r = cl[col].rank(pct=True) * 100
    pct[label] = r if hib else 100 - r


def radar(names):
    fig = go.Figure()
    colors = [ACCENT, "#ff6b6b"]
    for i, name in enumerate(names):
        idx = cl.index[cl["player"] == name][0]
        vals = pct.loc[idx].tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=list(AXES) + [list(AXES)[0]], fill="toself",
            name=name, line=dict(color=colors[i]), opacity=0.75,
        ))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], showticklabels=False)),
                      showlegend=len(names) > 1)
    return style_fig(fig, 460)


st.markdown(
    """<div class="hero"><h1>🏏 Cricket Scout AI</h1>
<p>Context-adjusted batting analytics built from every IPL delivery since 2008.
Find real talent, not lucky seasons.</p></div>""",
    unsafe_allow_html=True,
)

tab_over, tab_scout, tab_compare, tab_map, tab_how = st.tabs(
    ["Leaderboard", "Scouting report", "Compare players", "Style map", "How it works"]
)

# ---------------- Leaderboard ----------------
with tab_over:
    c = st.columns(4)
    card(c[0], "1,243", "IPL matches")
    card(c[1], "295,732", "Deliveries analysed")
    card(c[2], f"{len(cl)}", "Batters profiled")
    card(c[3], "5", "Batting styles found")
    st.write("")

    f1, f2 = st.columns(2)
    season = f1.selectbox("Season", ["All seasons"] + sorted(rae["year"].unique(), reverse=True))
    min_balls = f2.slider("Minimum balls faced", 60, 500, 250, step=20)

    view = rae[rae["balls"] >= min_balls]
    if season != "All seasons":
        view = view[view["year"] == season]
    top = view.sort_values("rae_per100", ascending=False).head(12).copy()
    top["label"] = top["player"] + " (" + top["year"].astype(str) + ")"

    fig = px.bar(
        top.iloc[::-1], x="rae_per100", y="label", orientation="h",
        color="rae_per100", color_continuous_scale=["#0f5e55", "#22d3a6"],
        text=top.iloc[::-1]["rae_per100"].round(1),
        labels={"rae_per100": "Runs above expected per 100 balls", "label": ""},
    )
    fig.update_traces(textposition="outside")
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(style_fig(fig, 500), use_container_width=True)
    st.caption("Runs above expected compares what a batter scored with what a typical batter "
               "would score in the same situation: over, wickets lost and chase pressure.")

# ---------------- Scouting report ----------------
with tab_scout:
    player = st.selectbox("Pick a batter", sorted(cl["player"]),
                          index=sorted(cl["player"]).index("V Kohli") if "V Kohli" in set(cl["player"]) else 0)
    row = cl[cl["player"] == player].iloc[0]
    p = pct.loc[row.name]
    hist = rae[(rae["player"] == player) & (rae["balls"] >= 60)].sort_values("year")

    best = p.sort_values(ascending=False)
    st.markdown(
        f'<div class="verdict"><b>{player}</b> profiles as a <b>{row["style"]}</b>. '
        f'Strongest trait: <b>{best.index[0].lower()}</b> ({best.iloc[0]:.0f}th percentile). '
        f'Weakest: <b>{best.index[-1].lower()}</b> ({best.iloc[-1]:.0f}th percentile). '
        f'Based on {int(row["balls"]):,} balls faced.</div>',
        unsafe_allow_html=True,
    )

    c = st.columns(4)
    card(c[0], f'{row["rel_sr"]:.2f}x', "Strike rate vs league")
    card(c[1], f'{row["six_pct"]:.1f}%', "Balls hit for six")
    card(c[2], f'{row["dot_pct"]:.1f}%', "Dot balls")
    card(c[3], f'{row["balls_per_out"]:.0f}', "Balls per dismissal")
    st.write("")

    l, r = st.columns([1, 1.2])
    with l:
        st.markdown("**Skill profile** (percentile vs 140 IPL batters)")
        st.plotly_chart(radar([player]), use_container_width=True)
    with r:
        st.markdown("**Season by season: runs above expected per 100 balls**")
        fig = px.bar(hist, x="year", y="rae_per100", color=hist["rae_per100"] > 0,
                     color_discrete_map={True: ACCENT, False: "#ff6b6b"},
                     hover_data=["balls", "runs"], labels={"rae_per100": "", "year": ""})
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 460), use_container_width=True)
        st.caption("Single seasons are noisy: only about 12% of a batter's performance carries "
                   "over to the next year. Look for consistent patterns, not one big bar.")

    z = (cl[[v[0] for v in AXES.values()]] - cl[[v[0] for v in AXES.values()]].mean()) / \
        cl[[v[0] for v in AXES.values()]].std()
    dist = ((z - z.loc[row.name]) ** 2).sum(axis=1) ** 0.5
    sim = cl.assign(distance=dist).drop(index=row.name).nsmallest(5, "distance")
    st.markdown("**Most similar batters**")
    st.dataframe(
        sim[["player", "style", "rel_sr", "six_pct", "dot_pct", "balls_per_out"]]
        .rename(columns={"rel_sr": "SR vs league", "six_pct": "Six %", "dot_pct": "Dot %",
                         "balls_per_out": "Balls/out"}).round(2),
        use_container_width=True, hide_index=True,
    )

# ---------------- Compare ----------------
with tab_compare:
    names = sorted(cl["player"])
    a, b = st.columns(2)
    p1 = a.selectbox("Player A", names, index=names.index("V Kohli") if "V Kohli" in names else 0)
    p2 = b.selectbox("Player B", names, index=names.index("MS Dhoni") if "MS Dhoni" in names else 1)
    st.plotly_chart(radar([p1, p2]), use_container_width=True)
    cols = ["player", "style", "rel_sr", "four_pct", "six_pct", "dot_pct", "balls_per_out", "pp_share", "death_share"]
    st.dataframe(cl[cl["player"].isin([p1, p2])][cols].round(2), use_container_width=True, hide_index=True)

# ---------------- Style map ----------------
with tab_map:
    st.markdown("Every dot is a batter. **Closer together means more similar playing style.** "
                "Dot size shows how many balls they have faced.")
    d = cl.copy()
    d["label"] = np.where(d["balls"].rank(ascending=False) <= 22, d["player"], "")
    fig = px.scatter(
        d, x="pc1", y="pc2", color="style", size="balls", text="label", hover_name="player",
        color_discrete_map=STYLE_COLORS,
        hover_data={"pc1": False, "pc2": False, "label": False, "rel_sr": ":.2f", "six_pct": ":.1f"},
        labels={"pc1": "", "pc2": ""},
    )
    fig.update_traces(textposition="top center", textfont=dict(size=11))
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)
    st.plotly_chart(style_fig(fig, 640), use_container_width=True)
    st.caption("The styles overlap at the edges. Batting is a spectrum, so these are useful "
               "descriptions, not hard categories (silhouette score 0.22).")
    prof = cl.groupby("style")[["rel_sr", "four_pct", "six_pct", "dot_pct", "balls_per_out",
                                "pp_share", "death_share"]].mean().round(2)
    st.dataframe(prof, use_container_width=True)

# ---------------- How it works ----------------
with tab_how:
    st.markdown(
        """
### The problem
Scouts judge batters on strike rate and average, but a single season is mostly luck, and a
130 strike rate in 2012 is not the same as a 130 in 2025.

### What I built
1. **Data pipeline.** Loaded 1,243 IPL matches (295,732 deliveries) from Cricsheet, and found
   and fixed a bug that silently duplicated every delivery (caught by checking Kohli's total
   against reality).
2. **Expected-runs model.** A gradient-boosted model predicts runs on each delivery from the
   match situation only (over, wickets, chase pressure), with no information about who is
   batting. A batter's *runs above expected* is how much they beat that benchmark.
3. **Style clustering.** K-means on era-adjusted features (scoring rate, boundaries,
   dot balls, survival, phase usage) found five batting styles.
4. **Honest evaluation.** Everything is tested on later seasons than it was trained on.

### What I found (including what did not work)
- A model predicting next-season strike rate from past seasons beat a "guess the league
  average" baseline by only **1%** (MAE 0.1046 vs 0.1057). Season results are too noisy.
- Season-to-season correlation is only **r = 0.35**, so about 12% of performance carries over.
- Runs above expected (r = 0.355) was about as persistent as strike rate (r = 0.343),
  so the context adjustment did not make talent easier to predict, but it does make
  comparisons across eras and match situations fairer.

### Tech
Python, pandas, scikit-learn, Plotly, Streamlit
"""
    )