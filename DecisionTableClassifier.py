import pandas as pd
from collections import Counter

class DecisionTableClassifier:
    def __init__(self):
        self.decision_table = {}
        self.target_column = None

    def load_data(self, file_path, target_column):
        """
        Load data from a CSV file and specify the target column.
        """
        self.data = pd.read_csv(file_path)
        self.target_column = target_column

    def build_decision_table(self):
        """
        Build the decision table based on majority class for each attribute combination.
        """
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in the data.")

        # Group data by all feature columns except the target
        feature_columns = self.data.columns.drop(self.target_column)
        grouped = self.data.groupby(list(feature_columns))

        # Create the decision table
        self.decision_table = {}
        for feature_values, group in grouped:
            # Determine the majority class in the target column
            majority_class = Counter(group[self.target_column]).most_common(1)[0][0]
            self.decision_table[tuple(feature_values)] = majority_class

    def predict(self, X):
        """
        Predict the class for a given set of features based on the decision table.
        """
        if not self.decision_table:
            raise ValueError("The decision table has not been built yet.")

        predictions = []
        for _, row in X.iterrows():
            feature_values = tuple(row)
            predictions.append(self.decision_table.get(feature_values, None))  # None if not found
        return predictions

    def evaluate(self, X, y):
        """
        Evaluate the classifier on a test dataset.
        """
        predictions = self.predict(X)
        correct = sum(1 for pred, true in zip(predictions, y) if pred == true)
        accuracy = correct / len(y)
        return accuracy

    def print_decision_table(self):
        """
        Print the decision table in a readable format.
        """
        if not self.decision_table:
            print("The decision table is empty. Build the table first.")
            return
        
        print("Decision Table:")
        print("-" * 50)
        print("{:<30} | {:<10}".format("Feature Values", "Majority Class"))
        print("-" * 50)
        for feature_values, majority_class in self.decision_table.items():
            formatted_values = ", ".join(map(str, feature_values))
            print("{:<30} | {:<10}".format(formatted_values, majority_class))
        print("-" * 50)
