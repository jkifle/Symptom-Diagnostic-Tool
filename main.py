import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# Load the training and testing data
training_data_path = 'Training.csv'
testing_data_path = 'Testing.csv'

training_data = pd.read_csv(training_data_path)
testing_data = pd.read_csv(testing_data_path)

# Load the training and testing data 

X_train = training_data.drop(columns=['prognosis'])
y_train = training_data['prognosis']
X_test = testing_data.drop(columns=['prognosis'])
y_test = testing_data['prognosis']



# Train the decision tree model
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

# Function to update disease probabilities based on user's answers
def update_disease_probabilities(disease_probs, symptom, response):
    for disease in disease_probs:
       #  general_risk = 0
        # Adjust probabilities based on symptom response
        
        if response == 'yes' and X_train[symptom].sum() > 0:
            # general_risk += 1
            if symptom in X_train.columns:
                symptom_positive_prob = X_train[y_train == disease][symptom].mean()
                disease_probs[disease] *= symptom_positive_prob
        elif response == 'no' and X_train[symptom].sum() > 0:
            if symptom in X_train.columns:
                symptom_negative_prob = 1 - X_train[y_train == disease][symptom].mean()
                disease_probs[disease] *= symptom_negative_prob
    return disease_probs

# Function to interactively ask questions and narrow down the disease
def interactive_diagnosis(clf, X_train):
    print("Welcome to the Interactive Healthcare Diagnostic Tool!")
    disease_probs = {disease: 1.0 for disease in clf.classes_}
    symptoms_asked = set()
    
    while True:
        # Find the next most informative symptom to ask about
        remaining_symptoms = [symptom for symptom in X_train.columns if symptom not in symptoms_asked]
        if not remaining_symptoms:
            break
        next_symptom = np.random.choice(remaining_symptoms)
        
        # Ask the user about the symptom
        response = input(f"Do you have {next_symptom.replace('_', ' ')}? (yes/no): ").strip().lower()
        symptoms_asked.add(next_symptom)
        
        # Update disease probabilities based on response
        disease_probs = update_disease_probabilities(disease_probs, next_symptom, response)
        
        # Normalize probabilities for comparison
        total_prob = sum(disease_probs.values())
        disease_probs = {disease: prob / total_prob for disease, prob in disease_probs.items()}
        
        # Show the current top disease predictions
        sorted_diseases = sorted(disease_probs.items(), key=lambda x: x[1], reverse=True)
        print("\nCurrent likely conditions:")
        for disease, probability in sorted_diseases[:3]:  # Show top 3 diseases
            print(f" - {disease}: {probability:.2f}")
          
        
        # Stop if we have a high-confidence diagnosis
        if sorted_diseases[0][1] > 0.8:
            print(f"\nDiagnosis Suggestion: Based on your symptoms, you likely have {sorted_diseases[0][0]}.")
            break
    
    print("Thank you for using the diagnostic tool. Please consult a healthcare professional for confirmation.")

# Run the interactive diagnosis tool
interactive_diagnosis(clf, X_train)