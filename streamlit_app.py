# ============================
#   STREAMLIT DASHBOARD (CLEAN)
# ============================

import streamlit as st
import requests
import time
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Complaint Pro — Live Dashboard", layout="wide")
st.title("📢 LIVE COMPLAINT MONITORING SYSTEM")

# -------------------------
# INITIAL VARIABLES
# -------------------------
neg_words = []
cat_count = {"billing": 0, "support": 0, "product": 0, "service": 0, "account": 0}
pos = neg = neu = 0

# Load sample data (your processed parquet file)
df = pd.read_parquet("complaintpro/data/processed/complaints.parquet").sample(20)

# UI containers
live = st.empty()
charts = st.empty()
wc = st.empty()
final = st.empty()

# -------------------------
# MAIN LOOP
# -------------------------
for i, txt in enumerate(df["clean_text"]):

    try:
        r = requests.post("http://localhost:8000/predict", json={"text": txt})
        d = r.json()

        category = d["category"]
        sentiment = d["sentiment"].lower()
        short_text = d["text"]

        live.success(f"{i+1}/20 → {category.upper()} | {sentiment.upper()} | {short_text}...")

        # Update counts
        cat_count[category] += 1

        if "neg" in sentiment:
            neg += 1
            neg_words.append(short_text)
        elif "pos" in sentiment:
            pos += 1
        else:
            neu += 1

        # -------------------------
        #   LIVE CHARTS
        # -------------------------
        with charts.container():
            col1, col2 = st.columns(2)

            # Category chart
            col1.plotly_chart(
                px.bar(
                    x=list(cat_count.keys()),
                    y=list(cat_count.values()),
                    title="Live Category Count",
                    color_discrete_sequence=["#636EFA"]
                ),
                use_container_width=True
            )

            # Sentiment chart
            col2.plotly_chart(
                px.pie(
                    values=[pos, neg, neu],
                    names=["POSITIVE", "NEGATIVE", "NEUTRAL"],
                    title="Live Sentiment",
                    color_discrete_sequence=["#00CC96", "#FF4444", "#636EFA"]
                ),
                use_container_width=True
            )

        # -------------------------
        # WORDCLOUD
        # -------------------------
        if len(neg_words) > 2:
            cloud = WordCloud(
                width=400,
                height=300,
                background_color="black",
                colormap="Reds"
            ).generate(" ".join(neg_words))

            fig, ax = plt.subplots()
            ax.imshow(cloud)
            ax.axis("off")
            wc.pyplot(fig)

        time.sleep(1.2)

    except Exception as e:
        live.error(f"API Offline: {e}")
        time.sleep(2)

# -------------------------
# FINAL SUMMARY
# -------------------------
with final.container():
    st.balloons()
    st.success("🎉 Analysis Complete — 20 Complaints Processed")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total", 20)
    col2.metric("Positive", pos)
    col3.metric("Negative", neg)

    st.subheader("Final Category Distribution")
    st.plotly_chart(px.bar(x=list(cat_count.keys()), y=list(cat_count.values())))

    st.subheader("Final Sentiment Distribution")
    st.plotly_chart(
        px.pie(values=[pos, neg, neu],
               names=["POSITIVE", "NEGATIVE", "NEUTRAL"],
               color_discrete_sequence=["#00CC96","#FF4444","#636EFA"])
    )
