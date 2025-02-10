# Explaining Recommendation Fairness from a User/Item Perspective

Recommender systems play a crucial role in personalizing user experiences, yet ensuring fairness in their outcomes remains an elusive challenge. This work explores the impact of individual users or items on the fairness of recommender systems, thus addressing a significant knowledge gap in the field. We introduce an innovative approach called Adding-Based Counterfactual Fairness Reasoning (ACFR), designed to elucidate recommendation fairness from the unique perspectives of users and items. Conventional methodologies, like erasing-based counterfactual analysis, pose limitations, particularly in modern recommender systems dealing with a large number of users and items. These traditional methods, by excluding specific users or items, risk disrupting the crucial relational structure central to collaborative filtering recommendations. In contrast, ACFR employs an adding-based counterfactual analysis, a unique strategy allowing us to consider potential, yet-to-happen user-item interactions. This strategy preserves the core user-item relational structure, while predicting future behaviors of users or items. The commonly used feature-based counterfactual analysis, relying on gradient-based optimization to identify interference on each feature, is not directly applicable in our case. In the recommendation scenario we consider, only interactions between users and items are present during model training—no distinct features are involved. Consequently, the traditional mechanism proves impractical for identifying interference on these existing interactions. Our extensive experiments validate the superiority of ACFR over traditional baseline methods, demonstrating significant improvements in recommendation fairness on benchmark datasets. This work, therefore, provides a fresh perspective and a promising methodology for enhancing fairness in recommender systems.

## Installation & Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/jasonshere/ACFR.git
   cd ACFR
   ```

2. **Create a Python 3.10 Virtual Environment**

   ```bash
   python3.10 -m venv .venv
   ```

3. **Activate the Virtual Environment**

   ```bash
   # Linux/Mac
   source .venv/bin/activate
   
   # Windows
   .venv\Scripts\activate
   ```

4. **Install Dependencies**

   ```bash
   pip3 install -r requirements.txt
   ```

---

## Usage

1. **Cross Validation Sets**

   ```bash
   python3 cross_validation.py
   ```
   This script creates cross-validation splits from your dataset.

2. **Train Base Model**

   ```bash
   python3 train_base_model.py
   ```
   Trains a baseline recommender model.

3. **Train Imputation Model**

   ```bash
   # DeepDM
   python3 train_imputation_deepdm.py
   
   # or FNN
   python3 train_imputation_fnn.py

   # or FWFMs
   python3 train_imputation_fwfms.py
   ```
   Each script trains a different imputation model.

4. **Minimize User/Item Unfairness**

   ```bash
   # For items
   python3 find_item_by_minimizing_item_unfairness.py

   # For users
   python3 find_user_by_minimizing_user_unfairness.py
   ```
   These scripts optimize fairness objectives, yielding user or item scores.

After these steps, you’ll find fairness-related scores for both users and items.

---

## Citation

If you use this work in your research, please cite:

```
@article{10.1145/3698877,
  author = {Li, Jie and Ren, Yongli and Sanderson, Mark and Deng, Ke},
  title = {Explaining Recommendation Fairness from a User/Item Perspective},
  year = {2024},
  issue_date = {January 2025},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {43},
  number = {1},
  issn = {1046-8188},
  url = {https://doi.org/10.1145/3698877},
  doi = {10.1145/3698877},
  abstract = {Recommender systems play a crucial role ... (abbreviated for brevity) ...,
  journal = {ACM Trans. Inf. Syst.},
  month = nov,
  articleno = {17},
  numpages = {30},
  keywords = {Explainable Fairness, Counterfactual Reasoning, Recommender Systems}
}
```

---

**Enjoy exploring and improving fairness in recommender systems with ACFR!**
