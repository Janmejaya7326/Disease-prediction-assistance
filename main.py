import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Reading the dataset and extracting symptom names
data = pd.read_csv("Disease.csv").dropna(axis=1)
symptom_columns = data.columns[:-1]  # All columns except the last one (prognosis)

# Checking the balance of the dataset
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})

# Encoding the target variable using LabelEncoder
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

# Splitting the data into training and testing sets
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Model building and evaluation with hyperparameter tuning
models = {
    "SVC": GridSearchCV(SVC(), {'kernel': ['linear', 'rbf'], 'C': [1, 10]}, cv=10),
    "Gaussian NB": GaussianNB(),
    "Random Forest": GridSearchCV(RandomForestClassifier(random_state=18), {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}, cv=10)
}

# Function for cross-validation scoring
def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))

# Producing cross-validation score for the models
cross_val_scores = {}
for model_name in models:
    model = models[model_name]
    model.fit(X_train, y_train)
    
    # Performing 10-fold cross-validation
    scores = cross_val_score(model, X, y, cv=10, n_jobs=-1, scoring=cv_scoring)
    cross_val_scores[model_name] = scores

# Function to predict disease based on symptoms
def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    input_data = [0] * len(X.columns)
    for symptom in symptoms:
        symptom = symptom.strip()
        if symptom in X.columns:
            input_data[X.columns.get_loc(symptom)] = 1
    
    input_data = pd.DataFrame([input_data], columns=X.columns)
    
    rf_prediction = final_rf_model.predict(input_data)[0]
    nb_prediction = final_nb_model.predict(input_data)[0]
    svm_prediction = final_svm_model.predict(input_data)[0]
    
    predictions_list = [rf_prediction, nb_prediction, svm_prediction]
    final_prediction = int(mode(predictions_list)[0])
    
    # Decode the predicted disease using the encoder
    predicted_disease = encoder.inverse_transform([final_prediction])[0]
    
    predictions = {
        "rf_model_prediction": encoder.inverse_transform([rf_prediction])[0],
        "naive_bayes_prediction": encoder.inverse_transform([nb_prediction])[0],
        "svm_model_prediction": encoder.inverse_transform([svm_prediction])[0],
        "final_prediction": encoder.inverse_transform([final_prediction])[0],
        "predicted_disease": predicted_disease
    }
    
    return predictions

# Function to validate symptoms
def validate_symptoms(input_symptoms, valid_symptoms):
    symptoms = [symptom.strip() for symptom in input_symptoms.split(",")]
    invalid_symptoms = [symptom for symptom in symptoms if symptom not in valid_symptoms]
    return invalid_symptoms

# Function to predict disease from GUI input and show results
def predictDiseaseFromGUI():
    # Get symptoms input from the user
    symptoms = symptoms_entry.get()
    
    # Validate the symptoms
    invalid_symptoms = validate_symptoms(symptoms, symptom_columns)
    if invalid_symptoms:
        messagebox.showerror("Invalid Symptoms", f"Invalid symptoms entered: {', '.join(invalid_symptoms)}\nPlease enter valid symptoms.")
        return
    
    # Perform prediction based on the input symptoms
    predictions = predictDisease(symptoms)
    
    # Display only the final predicted disease
    prediction_message = f"Predicted Disease: {predictions['predicted_disease']}"
    messagebox.showinfo("Predicted Disease", prediction_message)
    
    # Ask if the user wants to see predictions of all models
    show_all_models = messagebox.askyesno("Show All Models", "Do you want to see predictions of all three models?")
    if show_all_models:
        showModelPredictions(predictions)

# Function to display predictions of all models
def showModelPredictions(predictions):
    model_predictions = f"Random Forest Prediction: {predictions['rf_model_prediction']}\n"
    model_predictions += f"Gaussian NB Prediction: {predictions['naive_bayes_prediction']}\n"
    model_predictions += f"SVM Prediction: {predictions['svm_model_prediction']}\n"
    
    messagebox.showinfo("Model Predictions", model_predictions)

# Function to display all cross-validation and confusion matrix graphs
def display_all_graphs():
    # Display the cross-validation results
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")
    # Plotting the distribution of diseases
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Disease", y="Counts", data=temp_df)
    plt.xticks(rotation=90)
    plt.title("Distribution of Diseases in the Dataset")
    plt.xlabel("Disease")
    plt.ylabel("Count")
    plt.savefig("balance_dataset.png")
    plt.show()

    for model_name in models:
        model = models[model_name]
        model.fit(X_train, y_train)
        scores = cross_val_scores[model_name]
        print("=" * 60)
        print(model_name)
        print(f"Scores: {scores}")
        print(f"Mean Score: {np.mean(scores)}")
        print(f"Standard Deviation: {np.std(scores)}")

        # Plotting boxplot of scores for visualization
        plt.figure(figsize=(8, 6))
        plt.boxplot(scores, vert=False)
        plt.title(f"{model_name} - Cross Validation Scores")
        plt.xlabel("Accuracy")
        plt.show()
    
    # Display the confusion matrix for each model
    for model_name, model in models.items():
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cf_matrix = confusion_matrix(y_test, preds)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')
        plt.title(f"Confusion Matrix for {model.__class__.__name__}")
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()
    
    # Display the combined model's confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(cf_matrix_combined, annot=True, cmap='Blues', fmt='g')
    plt.title("Confusion Matrix for Combined Model on Test Dataset")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    print(f"Accuracy on Test dataset by the combined model: {acc_combined * 100:.2f}%")

# Final model training and testing on whole dataset
final_svm_model = SVC(kernel='linear', C=1)
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=18)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

# Reading the test data
test_data = pd.read_csv("Disease.csv").dropna(axis=1)
test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])

# Making predictions on test data using final models
svm_preds = final_svm_model.predict(test_X)
nb_preds = final_nb_model.predict(test_X)
rf_preds = final_rf_model.predict(test_X)

# Initialize an empty list for final predictions
final_preds = []

# Iterate over predictions from each model and compute mode
for i, j, k in zip(svm_preds, nb_preds, rf_preds):
    mode_result = mode([i, j, k])
    combined_pred = mode_result.mode.item()
    final_preds.append(combined_pred)

# Calculate accuracy and confusion matrix for combined predictions
acc_combined = accuracy_score(test_Y, final_preds)
cf_matrix_combined = confusion_matrix(test_Y, final_preds)

# Create a tkinter GUI window
root = tk.Tk()
root.title("Disease Prediction Assistance")
root.geometry("800x600")
root.configure(bg="lightblue")  # Set the background color of the window

# Create a heading
heading = tk.Label(root, text="Disease Prediction Assistance", font=("Arial", 24), fg="red", bg="lightblue")
heading.pack(pady=20)

# Create a frame for input and buttons
frame = tk.Frame(root, bg="lightblue")
frame.pack(pady=10)

# Create a label for symptoms input
symptoms_label = tk.Label(frame, text="Enter Symptoms (comma-separated):", font=("Arial", 14), bg="lightblue")
symptoms_label.grid(row=0, column=0, padx=10, pady=10)

# Create an entry widget to input symptoms
symptoms_entry = tk.Entry(frame, width=50, font=("Arial", 14), bg="white", fg="black")
symptoms_entry.grid(row=0, column=1, padx=10, pady=10)

# Create a button to trigger disease prediction
predict_button = tk.Button(frame, text="Predict Disease", command=predictDiseaseFromGUI, font=("Arial", 14), bg="green", fg="white")
predict_button.grid(row=0, column=2, padx=10, pady=10)

# Create a button to display all graphs
display_button = tk.Button(root, text="Model Performance", command=display_all_graphs, font=("Arial", 14), bg="orange", fg="white")
display_button.pack(pady=20)

# Run the tkinter event loop
root.mainloop()
