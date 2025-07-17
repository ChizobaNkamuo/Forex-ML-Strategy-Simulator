import pandas as pd
import src.feature_engineering as feature_engineering
import src.model as model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

window = 10
features_columns = ["Open", "Close", "Volume", "rsi"]
data = None 
features, targets = feature_engineering.create_flattened_features(data[features_columns], window)

split = int(0.8 * len(features))
features_train, features_test = features[:split], features[split:]
targets_train, targets_test = targets[:split], targets[split:]

classifier = RandomForestClassifier()
classifier.fit(features_train, targets_train)
predictions = classifier.predict(features_test)
cnf_matrix = confusion_matrix(targets_test, predictions)

print(cnf_matrix)
print(accuracy_score(targets_test, predictions))

flat_feature_names = [f"{col}_t-{window - i}" for i in range(window) for col in features_columns]
model.permutation_test(classifier, flat_feature_names, features_test, targets_test)

