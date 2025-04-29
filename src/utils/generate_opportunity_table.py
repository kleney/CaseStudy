# ============================================================
# Function to match discovered topics to business opportunities 
# and recommend priorities based on size and growth.
#   - High size + High growth = High Priority
#   - High size + Low growth = Medium Priority
#   - Low size + High growth = Medium Priority
#   - Low size + Low growth = Low Priority
#
# Now supports using externally generated clean_labels mapping
# to populate the "Trend" column.
# Author: Katharine Leney, April 2025 (updated)
# ============================================================

import pandas as pd

def generate_opportunity_table(topic_model, topics_over_time_df, clean_labels=None, top_n_topics=None):
    """
    Parameters:
    - topic_model: trained topic model (with or without cleaned labels applied).
    - topics_over_time_df: DataFrame with columns ["Topic", "Frequency", ...] over time.
    - clean_labels: optional dict mapping topic_id -> cleaned label (from generate_clean_labels).
    - top_n_topics: if provided, limit to the top N topics by count.
    Returns:
    - matched_df: DataFrame of matched business opportunities with Trend set to cleaned label (if provided).
    - unmatched_topics: list of topic labels that did not match any business opportunity.
    """
    # -------------------------------------------------
    # Define business opportunity mappings 
    # -------------------------------------------------
    business_opportunities = {
        "Safety": {"Implication": "Ongoing emphasis on flight and operational safety across the industry",
                   "Opportunity": "Invest in IOSA certification, risk management, and proactive safety programs"},
        "Security": {"Implication": "Need for secure and seamless passenger identity and screening",
                     "Opportunity": "Adopt biometric identity systems and enhance cybersecurity protocols"},
        "Settlement": {"Implication": "Evolution of industry financial frameworks (e.g. BSP, CASS)",
                       "Opportunity": "Upgrade payment systems and optimize cash flow through digital settlement"},
        "Charges": {"Implication": "Rising airport and regulatory charges impacting airline margins",
                    "Opportunity": "Advocate for transparency and collaborate on cost-efficiency initiatives"},
        "Global": {"Implication": "Macro-level trends influencing aviation growth and resilience",
                   "Opportunity": "Adapt strategy to respond to geopolitical, economic, and demographic shifts"},
        "New Distribution Capability": {"Implication": "Digital transformation of airline retail and distribution",
                                        "Opportunity": "Implement NDC to personalise offers and control distribution costs"},
        "Baggage": {"Implication": "Passenger expectations for efficient baggage services",
                     "Opportunity": "Introduce real-time tracking and self-service baggage solutions"},
        "Cargo": {"Implication": "Cargo remains a strong revenue pillar post-pandemic",
                   "Opportunity": "Expand dedicated cargo capacity and invest in digital freight platforms"},
        "SAF": {"Implication": "Industry push for sustainable fuels to meet net-zero goals",
                 "Opportunity": "Invest in SAF partnerships and explore supply chain integration"},
        "Markets": {"Implication": "Shifting demand patterns and regional recovery trends",
                     "Opportunity": "Refocus network planning on high-growth and underserved markets"},
        "Covid-19": {"Implication": "Operational disruption and long-term resilience planning",
                      "Opportunity": "Invest in flexible operating models and health safety protocols"},
        "Diversity & Inclusion": {"Implication": "Stakeholder expectations for diverse and inclusive leadership",
                                   "Opportunity": "Build diverse talent pipelines and track DEI performance"},
        "Data": {"Implication": "Explosion of data-driven use cases in aviation",
                       "Opportunity": "Adopt AI for predictive maintenance, dynamic pricing, and customer service"},
        "Training": {"Implication": "Need to attract and upskill talent amid workforce shortages",
                      "Opportunity": "Expand digital learning and modernise aviation training pathways"},
        "Taxation": {"Implication": "Growing tax pressures from governments on aviation activities",
                      "Opportunity": "Strengthen policy advocacy and evaluate operational tax exposures"},
        "Frequent Flyer": {"Implication": "Loyalty programs remain key to customer retention and data",
                             "Opportunity": "Modernise frequent flyer schemes and explore partnerships"},
        "Infrastructure": {"Implication": "Airports and systems under strain from demand and sustainability targets",
                            "Opportunity": "Partner on smart infrastructure projects and capacity investments"},
        "Digital Identity": {"Implication": "IATA and governments moving toward digital travel credentials",
                               "Opportunity": "Implement One ID and enhance digital onboarding processes"},
        "Lithium Batteries": {"Implication": "Safety concerns around transporting lithium batteries",
                                "Opportunity": "Strengthen dangerous goods training and handling procedures"},
        "Unruly Passengers": {"Implication": "Rising number of in-flight disruptions",
                                 "Opportunity": "Implement preventative policies and provide staff with de-escalation training"},
        "CEIV Programs": {"Implication": "Rising demand for certified pharma, fresh, and live cargo services",
                            "Opportunity": "Join CEIV programs and target high-value logistics niches"},
        "Workforce & Culture": {"Implication": "Industry-wide focus on employee wellbeing and agility",
                                  "Opportunity": "Create purpose-driven cultures and accelerate change adoption"},
    }

    # -------------------------------------------------
    # Extract topic info and counts
    # -------------------------------------------------
    topic_info = topic_model.get_topic_info()
    topic_df = topic_info.loc[topic_info["Topic"] != -1, ["Topic", "Name", "Count"]].copy()
    if top_n_topics is not None:
        topic_df = topic_df.nlargest(top_n_topics, "Count")
    total_count = topic_df["Count"].sum()

    # -------------------------------------------------
    # Compute topic growth
    # -------------------------------------------------
    growth_dict = {}
    for tid in topic_df["Topic"]:
        subset = topics_over_time_df[topics_over_time_df["Topic"] == tid]
        if subset.empty:
            growth = 0
        else:
            growth = subset.iloc[-1]["Frequency"] - subset.iloc[0]["Frequency"]
        growth_dict[tid] = growth

    # -------------------------------------------------
    # Match to business opportunities
    # -------------------------------------------------
    matched = []
    unmatched = []
    for _, row in topic_df.iterrows():
        tid = row["Topic"]
        raw_label = row["Name"]
        # Use cleaned label if provided, else fall back to raw
        trend_label = clean_labels.get(tid, raw_label) if clean_labels else raw_label
        size = row["Count"]
        frac = size / total_count
        growth = growth_dict.get(tid, 0)

        found = False
        for keyword, info in business_opportunities.items():
            if keyword.lower() in raw_label.lower():
                # Priority logic
                if frac > 0.02 and growth > 5:
                    prio = "High"
                elif (frac > 0.02 and growth > 0) or growth > 1:
                    prio = "Medium"
                else:
                    prio = "Low"

                matched.append({
                    "Trend": trend_label,
                    "Detected Label": raw_label,
                    "Implication": info["Implication"],
                    "Ongoing Activity": info["Opportunity"],
                    "Priority (Recommended)": prio,
                    "Keyword Fraction (%)": round(frac * 100, 2),
                    "Topic Growth": round(growth, 4)
                })
                found = True
                break
        if not found:
            unmatched.append(raw_label)

    # -------------------------------------------------
    # Final DataFrame and sorting
    # -------------------------------------------------
    df = pd.DataFrame(matched)
    if df.empty:
        print("\n WARNING: No matching business opportunities found for detected topics.")
        print("\t ==> Check if topic labels and business_opportunities mapping is aligned.\n")
    else:
        order = {"High": 0, "Medium": 1, "Low": 2, "To be assessed": 3}
        df["_prio"] = df["Priority (Recommended)"].map(order)
        df = df.sort_values(by=["_prio", "Keyword Fraction (%)"], ascending=[True, False])
        df = df.drop(columns=["_prio"]).reset_index(drop=True)

    return df, unmatched
