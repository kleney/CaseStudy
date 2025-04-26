import json
from bertopic import BERTopic
from tqdm import tqdm

# 1) Load chunks
with open("../data/chunks.json") as f:
    documents = json.load(f)

# 2) Fit topic model
topic_model = BERTopic(min_topic_size=10, nr_topics="auto")
topics, probs = topic_model.fit_transform(documents)

# 3) Inspect & save
freq = topic_model.get_topic_info()
topic_model.save("models/aviation_bertopic")

# 4) (Optional) export data for plotting
freq.to_csv("../data/topic_freq.csv", index=False)
