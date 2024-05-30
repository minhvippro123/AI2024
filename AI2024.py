from collections import defaultdict
import math


class NaiveBayesClassifier:
    def __init__(self):
        self.classes = defaultdict(int)
        self.feature_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.feature_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.class_probs = defaultdict(float)

    def train(self, data):
        # Count the occurrences of each class and feature values
        for record in data:
            label = record['Buy_Computer']
            self.classes[label] += 1
            for feature in record:
                if feature == 'Buy_Computer':
                    continue
                self.feature_counts[label][feature][record[feature]] += 1

        # Calculate probabilities
        total_records = len(data)
        for label in self.classes:
            self.class_probs[label] = self.classes[label] / total_records
            for feature in self.feature_counts[label]:
                total_feature_count = sum(self.feature_counts[label][feature].values())
                for value in self.feature_counts[label][feature]:
                    self.feature_probs[label][feature][value] = self.feature_counts[label][feature][
                                                                    value] / total_feature_count

    def predict(self, features):
        max_prob = -math.inf
        best_label = None

        for label in self.classes:
            log_prob = math.log(self.class_probs[label])
            for feature in features:
                if features[feature] in self.feature_probs[label][feature]:
                    log_prob += math.log(self.feature_probs[label][feature][features[feature]])
                else:
                    log_prob += math.log(1e-6)  # Smoothing for unseen features
            if log_prob > max_prob:
                max_prob = log_prob
                best_label = label

        return best_label


# Prepare the dataset
data = [
    {'Age': 'Young', 'Income': 'High', 'Student': 'No', 'Credit_Rating': 'Fair', 'Buy_Computer': 'No'},
    {'Age': 'Young', 'Income': 'High', 'Student': 'No', 'Credit_Rating': 'Excellent', 'Buy_Computer': 'No'},
    {'Age': 'Medium', 'Income': 'High', 'Student': 'No', 'Credit_Rating': 'Fair', 'Buy_Computer': 'Yes'},
    {'Age': 'Old', 'Income': 'Medium', 'Student': 'No', 'Credit_Rating': 'Fair', 'Buy_Computer': 'Yes'},
    {'Age': 'Old', 'Income': 'Low', 'Student': 'Yes', 'Credit_Rating': 'Fair', 'Buy_Computer': 'Yes'},
    {'Age': 'Old', 'Income': 'Low', 'Student': 'Yes', 'Credit_Rating': 'Excellent', 'Buy_Computer': 'No'},
    {'Age': 'Medium', 'Income': 'Low', 'Student': 'Yes', 'Credit_Rating': 'Excellent', 'Buy_Computer': 'Yes'},
    {'Age': 'Young', 'Income': 'Medium', 'Student': 'No', 'Credit_Rating': 'Fair', 'Buy_Computer': 'No'},
    {'Age': 'Young', 'Income': 'Low', 'Student': 'Yes', 'Credit_Rating': 'Fair', 'Buy_Computer': 'Yes'},
    {'Age': 'Old', 'Income': 'Medium', 'Student': 'Yes', 'Credit_Rating': 'Fair', 'Buy_Computer': 'Yes'},
    {'Age': 'Young', 'Income': 'Medium', 'Student': 'Yes', 'Credit_Rating': 'Excellent', 'Buy_Computer': 'Yes'},
    {'Age': 'Medium', 'Income': 'Medium', 'Student': 'No', 'Credit_Rating': 'Excellent', 'Buy_Computer': 'Yes'},
    {'Age': 'Medium', 'Income': 'High', 'Student': 'Yes', 'Credit_Rating': 'Fair', 'Buy_Computer': 'Yes'},
    {'Age': 'Old', 'Income': 'Medium', 'Student': 'No', 'Credit_Rating': 'Excellent', 'Buy_Computer': 'No'},
]

# Create and train the model
nb = NaiveBayesClassifier()
nb.train(data)

# Prompt the user to enter features
age = input("Enter Age (Young, Medium, Old): ")
income = input("Enter Income (Low, Medium, High): ")
student = input("Is the person a student? (Yes, No): ")
credit_rating = input("Enter Credit Rating (Fair, Excellent): ")

# Define the features of the new record
new_record = {
    'Age': age,
    'Income': income,
    'Student': student,
    'Credit_Rating': credit_rating
}

# Make a prediction
prediction = nb.predict(new_record)
print(f'The predicted class for the given features is: {prediction}')
