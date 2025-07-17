import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

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
