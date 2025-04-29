from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Expand list of stopwords to remove industry-specific noise
# and months of the year
iata_stopwords = [
    "sky", "flight", "air", "aviation", "iata", "sector", "industry",
    "annual", "review", "general", "director", "transport", "association",
    "chief", "executive", "officer", "bisignani", "giovanni", "ceo",
    "board", "governor", "governors", "chairman", "president", "mr", "ms", "miss", "mrs", "dr",
    "airlines", "airline", "business", "businesses", "airways", "airway",
    "representing", "representatives", "representative", "fax", "tel"
]
months = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]
month_abbr = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec"
]
CUSTOM_STOPWORDS = sorted(list(ENGLISH_STOP_WORDS.union(iata_stopwords, months, month_abbr)))
