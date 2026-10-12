"""Cricket Scout AI: interactive dashboard."""
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Cricket Scout AI", layout="wide")

FEATS = ["rel_sr", "four_pct", "six_pct", "dot_pct", "balls_per_out", "pp_share", "death_share"]


@st.cache_data
def load():
    rae = pd.read_parquet("data/processed/batter_rae.parquet")
    clusters = pd.read_parquet("data/processed/batter_clusters.parquet")
    return rae, clusters


rae, clusters = load()

st.title("Cricket Scout AI")
st.caption(
    "IPL batting analytics from 295,000 deliveries. Runs above expected adjusts each "
    "player's scoring for the match situation (over, wickets, chase pressure)."
)

tab1, tab2, tab3 = st.tabs(["Leaderboard", "Style map", "Player explorer"])

# ---------- Tab 1: leaderboard ----------
with tab1:
    c1, c2 = st.columns(2)
    year_choice = c1.selectbox("Season", ["All seasons"] + sorted(rae["year"].unique(), reverse=True))
    min_balls = c2.slider("Minimum balls faced", 60, 500, 200, step=20)

    view = rae[rae["balls"] >= min_balls]
    if year_choice != "All seasons":
        view = view[view["year"] == year_choice]

    top = view.sort_values("rae_per100", ascending=False).head(15)
    top = top.assign(label=top["player"] + " " + top["year"].astype(str))
    fig = px.bar(
        top.iloc[::-1], x="rae_per100", y="label", orientation="h",
        hover_data=["balls", "runs"], labels={"rae_per100": "Runs above expected per 100 balls", "label": ""},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Season-to-season, only about 12% of batting performance carries over (r = 0.35). "
        "Treat short seasons with caution."
    )

# ---------- Tab 2: style map ----------
with tab2:
    st.subheader("Batting styles")
    fig = px.scatter(
        clusters, x="pc1", y="pc2", color="style", size="balls", hover_name="player",
        hover_data={"pc1": False, "pc2": False, "rel_sr": ":.2f", "six_pct": ":.1f", "balls": True},
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        clusters.groupby("style")[FEATS].mean().round(2), use_container_width=True
    )

# ---------- Tab 3: player explorer ----------
with tab3:
    player = st.selectbox("Pick a player", sorted(clusters["player"]))
    row = clusters[clusters["player"] == player].iloc[0]
    st.markdown(f"### {player}: *{row['style']}*")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Strike rate vs league", f"{row['rel_sr']:.2f}x")
    m2.metric("Six %", f"{row['six_pct']:.1f}")
    m3.metric("Dot %", f"{row['dot_pct']:.1f}")
    m4.metric("Balls per dismissal", f"{row['balls_per_out']:.0f}")

    history = rae[(rae["player"] == player) & (rae["balls"] >= 60)]
    fig = px.line(
        history, x="year", y="rae_per100", markers=True,
        labels={"rae_per100": "Runs above expected per 100 balls"},
    )
    fig.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

    # Similar players: nearest neighbours on the standardised style features
    z = (clusters[FEATS] - clusters[FEATS].mean()) / clusters[FEATS].std()
    dist = ((z - z.loc[row.name]) ** 2).sum(axis=1) ** 0.5
    similar = clusters.assign(distance=dist).drop(index=row.name).nsmallest(5, "distance")
    st.markdown("**Most similar batters**")
    st.dataframe(similar[["player", "style", "rel_sr", "six_pct", "dot_pct"]].round(2),
                 use_container_width=True, hide_index=True)