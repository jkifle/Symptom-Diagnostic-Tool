# Symptom Diagnostic Tool

An interactive command-line tool that uses a decision tree classifier to predict possible diseases based on user-reported symptoms.

## Overview

This Python program guides users through a series of yes/no questions about their symptoms. 
It updates the probability of each disease based on user input and suggests the most likely 
condition when confidence is high.

This project was developed as part of the NSBE Region III FRC Impact-a-thon, a hackathon focused 
on creating tech solutions for underserved communities. We also got 2nd place for its innovation 
and potential healthcare impact.

## Features

- Decision Tree model trained on symptom-disease dataset
- Interactive symptom-based questioning
- Probability-based narrowing of possible conditions
- Displays top 3 most likely diagnoses after each input
- Early exit on high-confidence predictions

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn

Install dependencies with:
pip install pandas numpy scikit-learn

## How to Run
Ensure main.py, Training.csv, and Testing.csv are in the same directory. Then run:
python main.py
