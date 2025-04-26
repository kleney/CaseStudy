# ============================================================
# Function to match discovered topics to business opportunities 
# and recommend priorities based on size and growth.
#   - High size + High growth = High Priority
#   - High size + Low growth = Medium Priority
#   - Low size + High growth = Medium Priority
#   - Low size + Low growth = Low Priority
#
# Author: Katharine Leney, April 2025
# ============================================================

import pandas as pd

def generate_opportunity_table(topic_model, topics_over_time_df, top_n_topics=None):
    """
    Match BERTopic discovered topics to business opportunities and recommend priorities based on size and growth.

    Args:
        topic_model (BERTopic): A fitted BERTopic model.
        topics_over_time_df (DataFrame): Output from model.topics_over_time().
        top_n_topics (int, optional): Limit to top N topics by size.

    Returns:
        tuple: (matched_opportunities_df, unmatched_topics_list)
    """
    # -------------------------------------------------
    # Define business opportunity mappings 
    # (expand as needed!)
    # -------------------------------------------------
    business_opportunities = {
        "Growth": {
            "Implication": "Strong recovery in air travel post-pandemic",
            "Opportunity": "Expand network capacity and optimize route planning",
        },
        "Safety": {
            "Implication": "Continued industry focus on safety standards",
            "Opportunity": "Invest in IOSA certification and safety audits",
        },
        "Emissions": {
            "Implication": "Increasing pressure for environmental responsibility",
            "Opportunity": "Adopt Sustainable Aviation Fuels (SAF) and carbon offset programs",
        },
        "Airport": {
            "Implication": "Growing airport-related fees and costs",
            "Opportunity": "Negotiate airport charges and improve operational efficiency",
        },
        "Security": {
            "Implication": "Passenger data security and travel security concerns",
            "Opportunity": "Enhance digital identity management and passenger screening",
        },
        "Settlement": {
            "Implication": "Changes in financial settlement systems (BSP, etc.)",
            "Opportunity": "Optimize settlement systems and implement faster payment technologies",
        },
        "Cargo": {
            "Implication": "Continued strength in air cargo and freight demand",
            "Opportunity": "Expand cargo services and invest in logistics technology",
        },
        "New Distribution Capability": {
            "Implication": "Shift to direct airline distribution models",
            "Opportunity": "Adopt NDC-compliant sales and order management platforms",
        },
        "Regulation": {
            "Implication": "Tighter regulatory environment for passenger rights",
            "Opportunity": "Implement compliance monitoring and customer care improvements",
        },
        "China": {
            "Implication": "Strategic importance of Chinese and Asia-Pacific aviation markets",
            "Opportunity": "Develop partnerships and routes in China and APAC regions",
        },
        "Data": {
            "Implication": "Need for better data management and business intelligence",
            "Opportunity": "Invest in data analytics and AI-based decision systems",
        },
        "Tax": {
            "Implication": "Government pressure via taxes on aviation",
            "Opportunity": "Engage in advocacy and optimize tax planning strategies",
        },
        "Training": {
            "Implication": "Ongoing need for aviation training and upskilling",
            "Opportunity": "Expand training programs and digital learning platforms",
        },
        "Slot": {
            "Implication": "Airport slot scarcity impacting capacity",
            "Opportunity": "Optimize slot management and invest in slot trading systems",
        },
        "Board": {
            "Implication": "Board representation and governance changes",
            "Opportunity": "Strengthen board leadership and stakeholder engagement",
        },
        "Commercial": {
            "Implication": "Need for revenue diversification",
            "Opportunity": "Develop new commercial partnerships and ancillary revenue streams",
        },
        "Baggage": {
            "Implication": "Focus on passenger experience and baggage handling",
            "Opportunity": "Invest in baggage tracking technology and self-service processes",
        },
        "Diversity": {
            "Implication": "Industry-wide DEI (Diversity, Equity, Inclusion) pressure",
            "Opportunity": "Strengthen diversity and inclusion initiatives",
        },
        "ID": {
            "Implication": "Move towards digital identity and seamless travel",
            "Opportunity": "Implement biometric and digital ID solutions",
        },
        "Lithium": {
            "Implication": "Rising risks from transporting lithium batteries",
            "Opportunity": "Strengthen dangerous goods handling protocols",
        },
        "Fuel": {
            "Implication": "Volatility of jet fuel prices",
            "Opportunity": "Implement fuel hedging strategies and optimize fuel efficiency",
        },
        "Bar-Coded Boarding Pass": {
            "Implication": "Need for seamless, contactless boarding",
            "Opportunity": "Implement BCBP standards across operations",
        },
        "Unruly": {
            "Implication": "Rise in unruly passenger incidents",
            "Opportunity": "Enhance crew training and incident management systems",
        },
        "Center of Excellence for Independent Validators": {
            "Implication": "Importance of CEIV pharma and logistics certifications",
            "Opportunity": "Expand into certified logistics (e.g., pharma transport)",
        },
        "Business": {
            "Implication": "Organizational agility and change management",
            "Opportunity": "Simplify business processes and accelerate change initiatives",
        }
    }

    # -------------------------------------------------
    # Extract topic labels
    # -------------------------------------------------
    topic_info = topic_model.get_topic_info()
    # Exclude Topic = -1
    topic_labels = topic_info.loc[topic_info["Topic"] != -1, ["Topic", "Name", "Count"]]

    if top_n_topics is not None:
        topic_labels = topic_labels.nlargest(top_n_topics, "Count")

    # Total number of topics
    total_count = topic_labels["Count"].sum()   

    # -------------------------------------------------
    # Calculate growth rates from topics_over_time
    # ***VERY BASIC*** estimate, calculated as:
    #       latest_frequency - earliest_frequency
    # Could be improved later :-)
    # -------------------------------------------------
    growth_dict = {}

    for topic_id in topic_labels["Topic"]:
        topic_data = topics_over_time_df[topics_over_time_df["Topic"] == topic_id]
        if topic_data.empty:
            growth = 0
        else:
            # Simple growth: compare final frequency vs initial frequency
            growth = topic_data.iloc[-1]["Frequency"] - topic_data.iloc[0]["Frequency"]
        growth_dict[topic_id] = growth

    # -------------------------------------------------
    # Match detected topics to business opportunities
    # ------------------------------------------------- 
    matched_opportunities = []
    unmatched_topics = []

    for _, row in topic_labels.iterrows():
        topic_id = row["Topic"]
        label = row["Name"]
        size = row["Count"]
        fraction = size / total_count
        growth = growth_dict.get(topic_id, 0)

        matched = False
        for keyword, info in business_opportunities.items():
            if keyword.lower() in label.lower():
                # Recommend priority based on keyword fraction and growth
                #   - High keyword fraction + High growth = High Priority
                #   - High keyword fraction + Low growth = Medium Priority
                #   - Low keyword fraction + High growth = Medium Priority
                #   - Low keyword + Low growth = Low Priority
                if fraction > 0.1 and growth > 0.01:
                    priority = "High"
                elif fraction > 0.05 or growth > 0.01:
                    priority = "Medium"
                else:
                    priority = "Low"

                matched_opportunities.append({
                    "Trend": keyword,
                    "Detected Label": label,
                    "Implication": info["Implication"],
                    "Opportunity": info["Opportunity"],
                    "Priority (Recommended)": priority,
                    "Keyword Fraction (%)": round(fraction * 100, 2),
                    "Topic Growth": round(growth, 4)
                })
                matched = True
                break

        if not matched:
            unmatched_topics.append(label)

    # -------------------------------------------------
    # Output
    # -------------------------------------------------
    matched_df = pd.DataFrame(matched_opportunities)

    if matched_df.empty:
        print("\n WARNING: No matching business opportunities found for detected topics.")
        print("\t ==> Check if topic labels and business_opportunities mapping is aligned.\n")
    else:
        # Define custom priority ordering
        priority_order = {"High": 0, "Medium": 1, "Low": 2, "To be assessed": 3}

        # Map priority to a sortable number
        matched_df["Priority_Rank"] = matched_df["Priority (Recommended)"].map(priority_order)

        matched_df = matched_df.sort_values(
            by=["Priority_Rank", "Keyword Fraction (%)"],
            ascending=[True, False]  # Priority ascending (0,1,2), Fraction descending
        ).drop(columns=["Priority_Rank"]).reset_index(drop=True)

    return matched_df, unmatched_topics
