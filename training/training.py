import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

def preprocess_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    
    for col in df.columns[1:]:
        df[col] = df[col].apply(lambda x: tuple(map(float, x.strip('()').split(','))))

    
    labels = df['Label']
    features = df.drop(columns=['Label'])

    
    for i in range(1, 22):
        features[f'Keypoint_{i}_x'] = features[f'Keypoint_{i}'].apply(lambda x: x[0])
        features[f'Keypoint_{i}_y'] = features[f'Keypoint_{i}'].apply(lambda x: x[1])

    
    features.drop(columns=df.columns[1:22], inplace=True)

    return features, labels


def save_model(model, model_filename):
    joblib.dump(model, model_filename)

def train_model(features, labels):
    X = features
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    label_encoder_filename = '../label_encoder.joblib'
    save_model(label_encoder, label_encoder_filename)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_true = y_test
    y_pred = model.predict(X_test)
    report = classification_report(y_true, y_pred)
    print(report)
    return model

train_csv_path = '../train_data.csv'
test_csv_path = '../test_data.csv'

train_features, train_labels = preprocess_dataset(train_csv_path)
test_features, test_labels = preprocess_dataset(test_csv_path)

model = train_model(train_features, train_labels)
model_filename = '../trained_model.joblib'
save_model(model, model_filename)