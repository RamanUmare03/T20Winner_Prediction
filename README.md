# T20 Winner Prediction

## Role-Based Team Strengths for Cricket Match Prediction
Match-level feature engineering, leakage-safe joins, role analytics, and model benchmarking
TL;DR
Transforms past match-level player data (no heavy ball-by-ball) into role-based team strengths (openers, top/middle/lower order; pace/spin/all-round/part-time), visualizes how role imbalances drive wins, and benchmarks a model zoo. Results (test split): Bagging tops Accuracy/Balanced Accuracy (~0.695), SVC leads AUC (~0.688), LDA gives best probability calibration (LogLoss ~0.640, Brier ~0.225). All features are computed only from prior matches to avoid leakage.

## Models Used
The core of the prediction system is an ensemble model combining **XGBoost** and **LightGBM**. These models were chosen for their:

- **High accuracy** and **F1 score**.
- Ability to effectively manage **imbalanced datasets**, which is crucial in sports predictions where outcomes can be skewed.

## Feature Engineering
![debc5844-9002-4f62-8ba1-ea4dddbffeb7](https://github.com/user-attachments/assets/08da708a-46d6-4678-b28f-5d1283105685)

A key aspect of this project is the development of novel features that capture various factors influencing match outcomes. One such feature is the **`valuable_players_ratio`**, which assigns a score to each team based on the past performances of its players. This feature, along with others, enhances the model's predictive power by providing deeper insights into team dynamics.

## Model Performance
On unseen data, the model consistently achieves an accuracy of **65-70%**. This level of performance indicates a strong ability to generalize and provides valuable predictions for real-world applications.

## Conclusion
This project demonstrates the application of machine learning in sports analytics, offering a practical tool for predicting T20 match outcomes. The ensemble model's effectiveness and the innovative features developed make it a powerful resource for strategic planning and analysis.
