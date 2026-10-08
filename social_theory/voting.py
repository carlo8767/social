import os
import string
from tabulate import tabulate
import openml
import category_encoders as ce

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.calibration import CalibratedClassifierCV


#Loading the dataset #
def  load_dataset(id_dataset:int, target_label:string):
        dataset_uploade = openml.datasets.get_dataset(id_dataset)

        return dataset_uploade.get_data(target= target_label)

# Train/test split #
def split_data (x_values_features, y_label):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def convert_categorical(x_train, x_test):
    # Encode categorical features
    categorical_cols = x_train.select_dtypes(
        include=['category']
    ).columns

    encoder = ce.OrdinalEncoder(
        cols=categorical_cols
    )

    x_train = encoder.fit_transform(x_train)
    x_test = encoder.transform(x_test)

    return x_train, x_test ,encoder

def convert_original(x_train, original_encoder):
    return encoder.inverse_transform(X_train)


def print_benchmark_matrix(y_test_comparison, model, name):
   accuracy = format(accuracy_score (y_test_comparison, model), ".2f")
   precision = format(precision_score(y_test_comparison, model), ".2f")
   recall = format(recall_score(y_test_comparison, model), ".2f")
   table = [["MODEL", name], ["ACCURACY", accuracy],
            ["PRECISION", precision], ["RECALL", recall]]
   print(tabulate(table))


# Eventually use
api_key_open_ml = os.environ['API_KEY']

X, y, categorical_indicator, attribute_names = load_dataset(31, "class")
X_train, X_test, y_train, y_test = split_data(X, y)
X_train, X_test, encoder = convert_categorical(X_train, X_test)

# Encode target
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)



# Create models
model1 = LogisticRegression(  random_state=42,max_iter=500000)
model2 = DecisionTreeClassifier(random_state=42)
model3 = CalibratedClassifierCV(SVC(random_state=42), ensemble=False)


# Hard Voting Classifier and aggregation
hard_voting = VotingClassifier(
    estimators=[
        ('lr', model1),
        ('dt', model2),
        ('svc', model3)
    ],
    voting='hard'
)
hard_voting.fit(X_train, y_train)

# Individual predictions_model return nparray
lr_pred = hard_voting.named_estimators_['lr'].predict(X_test)
dt_pred = hard_voting.named_estimators_['dt'].predict(X_test)
svc_pred = hard_voting.named_estimators_['svc'].predict(X_test)

aggregate_hard_vote = hard_voting.predict(X_test)
# Convert eventually to original
original_label = label_encoder.inverse_transform(aggregate_hard_vote)

# Visualization single scores
print_benchmark_matrix(y_test, lr_pred , "Logistic Regression")
print_benchmark_matrix(y_test, dt_pred , "Decision Tree")
print_benchmark_matrix(y_test, svc_pred , "Support Vector")
print_benchmark_matrix(y_test, aggregate_hard_vote , "Hard Voting")

# https://www.kaggle.com/code/pythonafroz/votingclassifier-a-powerful-ensemble-technique