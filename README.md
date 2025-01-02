## Objective
To conduct a comprehensive analysis of social media posts related to H5N1 outbreaks using posts and comments from Reddit communities for various states in the US from early 2022 to mid 2024.

[Published Research Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5078277)

## Sources

- **Outbreaks Data:** [USDA HPAI Detections](https://www.aphis.usda.gov/livestock-poultry-disease/avian/avian-influenza/hpai-detections/commercial-backyard-flocks)
  
- **Reddit API Service:** [PullPush API](https://pullpush.io/)
  
- **Python Packages:** [PyPI](https://pypi.org/)
  
- **Sentiment Classification:** [BERT-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)

  - **Training Dataset:** [Emotions from Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/emotions)
  
- **Topic Modeling:**
  - [BERTopic](https://maartengr.github.io/BERTopic/index.html)
  - [Latent Dirichlet Allocation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)

## Requirements

- **Python Version:** `3.11.9`
- Install dependencies:
  
  ```bash
  pip install -r requirements.txt
  ```
