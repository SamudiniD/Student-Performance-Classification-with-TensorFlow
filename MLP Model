# Required Libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

# Load Dataset
url = 'https://raw.githubusercontent.com/lakminia/academic_data/main/academic_dataset.csv'
dataset = pd.read_csv(url)

# Data Preprocessing
dataset_encoded = pd.get_dummies(dataset, columns=['gender', 'NationalITy', 'PlaceofBirth', 'StageID',
                                                   'GradeID', 'SectionID', 'Topic', 'Semester',
                                                   'Relation', 'ParentAnsweringSurvey',
                                                   'ParentschoolSatisfaction', 'StudentAbsenceDays'],
                                                   drop_first=True)

scaler = MinMaxScaler()
numerical_features = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']
dataset_encoded[numerical_features] = scaler.fit_transform(dataset_encoded[numerical_features])

label_encoder = LabelEncoder()
dataset_encoded['Class'] = label_encoder.fit_transform(dataset['Class'])

X = dataset_encoded.drop(columns=['Class'])
y = dataset_encoded['Class']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Definition
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping])

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')

# Classification Report
y_pred = model.predict(X_val)
y_pred_classes = tf.argmax(y_pred, axis=1)
print(classification_report(y_val, y_pred_classes, target_names=['Low', 'Middle', 'High']))


