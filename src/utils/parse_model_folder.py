# ======================================================
# Helper function to parse the model folder name to 
# extract filter info
# Author: Katharine Leney, April 2025
# ======================================================

import re
import re

def parse_model_folder(folder_name):
    """
    Args:
        folder_name (str): e.g. 'bertopic_cluster_model_from_2015_excl_2020_2021_20250426'

    Returns:
        str: Human-readable description
    """
    description_parts = []

    # Remove the date suffix (always 8 digits at end)
    folder_name = re.sub(r'_\d{8}$', '', folder_name)

    # Extract from year
    from_match = re.search(r'from_(\d{4})', folder_name)
    if from_match:
        from_year = from_match.group(1)
        description_parts.append(f"From {from_year}")

    # Extract up_to year
    upto_match = re.search(r'up_to_(\d{4})', folder_name)
    if upto_match:
        upto_year = upto_match.group(1)
        description_parts.append(f"Up to {upto_year}")

    # Extract excluded years
    excl_match = re.search(r'excl_([\d_]+)', folder_name)
    if excl_match:
        years = excl_match.group(1).split("_")
        description_parts.append(f"Excluding {', '.join(years)}")

    return " | ".join(description_parts)
