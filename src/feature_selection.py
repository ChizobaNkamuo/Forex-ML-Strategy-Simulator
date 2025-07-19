import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

def permutation_test(model, feature_names, features_test, targets_test):
    result = permutation_importance(
        model, features_test, targets_test, n_repeats=10, random_state=1, n_jobs=2
    )
    forest_importances = pd.Series(result.importances_mean, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()

def correlation_analysis(data, features_columns):
    features_columns = features_columns[:]
    features_columns.append("log_return")
    correlations = data[features_columns].corr()["log_return"]
    print(correlations.sort_values(ascending=False))

    for feature in features_columns:
        rolling_corr = data[feature].rolling(window=100).corr(data["log_return"])
        print(f"{feature} correlation stability: {rolling_corr.std():.4f}")
    
def tree_analysis(features_columns, features_train,targets_train):
    random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
    random_forest.fit(features_train, targets_train)

    # Built-in importance (based on impurity decrease)
    importance_df = pd.DataFrame({
        "feature": features_columns,
        "importance": random_forest.feature_importances_
    }).sort_values("importance", ascending=False)
    print(importance_df)