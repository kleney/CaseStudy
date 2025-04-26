import json
from collections import defaultdict # counting by year and keyword
import pandas as pd
import matplotlib.pyplot as plt
import os
import re # regular expressions to extract years from filenames
import plotly.express as px

# ============================================================
#
# Simple script to count instances of keywords from a defined
# list, print in a simple table, and make a trend plot.
#
# ============================================================


# ------------------------------------------------------------
# Keywords to search for in the text
# ------------------------------------------------------------
KEYWORDS = [
    "sustainability", "biofuel", "net zero",
    "digital", "AI", "data", "cybersecurity",
    "regulation", "safety", "climate", "SAF",
    "economic"
]

# ------------------------------------------------------------
# Helper function to make text all lowercase
# (use with caution for acronyms!!)
# ------------------------------------------------------------
def normalise(text):
    return text.lower().replace("–", "-").strip()

# ------------------------------------------------------------
# Helper function to load chunks
# Returns a list of dicts like: { "year": ..., "text": ... }
# ------------------------------------------------------------
def load_chunks(path):
    with open(path) as f:
        return json.load(f)

# ------------------------------------------------------------
# Loop over all chunks, and for each one loop over all
# of the keywords and count the number of instances of each.
# ------------------------------------------------------------
def keyword_trends(chunks, keywords):
    yearly_counts = defaultdict(lambda: defaultdict(int))
    
    for chunk in chunks:
        year = chunk.get("year", "unknown")
        text = chunk["text"]

        for keyword in keywords:
            # Case-sensitive exact matches for "AI" and "SAF" to avoid false positives
            if keyword == "AI":
                count = len(re.findall(r"\bAI\b", text))
            elif keyword == "SAF":
                count = len(re.findall(r"\bSAF\b", text))
            else:
                # Case-insensitive full-word match for everything else
                count = len(re.findall(rf"\b{re.escape(keyword.lower())}\b", text.lower()))
            
            yearly_counts[year][keyword] += count

    return yearly_counts

# ------------------------------------------------------------
# Converts nested dictionary to pandas dataframe
# and does some cleaning
# ------------------------------------------------------------
def to_dataframe(yearly_counts):
    df = pd.DataFrame(yearly_counts)
    df = df.T # transpose
    df = df.fillna(0) # fill missing values with 0's
    df = df.astype(int) # make sure all values are integers
    return df.sort_index() # sort by year

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_path = os.path.join(ROOT, "data", "chunks.json")

    chunks = load_chunks(input_path)
    trends = keyword_trends(chunks, KEYWORDS)
    df = to_dataframe(trends)

    # Print simple table to terminal
    print("Keyword frequencies per year:")
    print(df)

    # Simple trend plot
    #df.plot(figsize=(12, 6), marker="o")
    #plt.title("Keyword Trends in IATA Annual Reviews")
    #plt.xlabel("Year")
    #plt.ylabel("Mentions")
    #plt.xticks(rotation=45)
    #plt.tight_layout()
    #plt.grid(False)
    #plt.show()

    # Melt the dataframe to long format for Plotly
    df_long = df.reset_index().melt(id_vars="index", var_name="Keyword", value_name="Mentions")
    df_long.rename(columns={"index": "Year"}, inplace=True)

    fig = px.line(
        df_long,
        x="Year",
        y="Mentions",
        color="Keyword",
        markers=True,
        title="Keyword Trends in IATA Annual Reviews (2005-2024)",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.D3
    )

    # Hacks to somewhat imitate IATA style
    fig.update_layout(
        width=1200,
        height=600,
        xaxis_title=None,
        yaxis_title=None,  
        annotations=[
            dict(
                xref='paper',
                yref='paper',
                x=-0.05,
                y=1.02,
                text="Mentions",
                showarrow=False,
                font=dict(size=14),
                xanchor='left',
                yanchor='bottom'
            )
        ],
        xaxis=dict(
            showgrid=False,
            showline=True,         
            linecolor='black',     
            linewidth=1,
            ticks="outside"
        ),
        yaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='black',
            linewidth=1,
            ticks="outside",
            range=[0, None]
        ),
        legend_title=None,
        #legend=dict(
        #    orientation="h",    # horizontal layout
        #    yanchor="bottom",
        #    y=1.08,             # position just above the chart
        #    xanchor="left",
        #    x=0
        #),
        hovermode="closest",
        font=dict(family="Arial", size=14),
        title_font=dict(size=20),
    )
    fig.write_image("outputs/simple_keyword_trends.png")
    fig.write_html("outputs/simple_keyword_trends.html")   
    fig.show()
    
