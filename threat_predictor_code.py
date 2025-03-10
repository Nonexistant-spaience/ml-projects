
import streamlit as st
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
import pandas as pd

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load or create SHAP Explainer
explainer = shap.TreeExplainer(model)

# Streamlit app
st.title('Threat Predictor 1.0')
st.write('Enter your input in the following fields')

age = st.number_input('Age:')

# Education Level
st.write("His education level")
ed_option = st.selectbox(
    "Choose an option:", 
    ["High School", "Bachelor", "Masters", "PhD"], 
    key="education_level"
)
ed_lvl = {"High School": 0, "Bachelor": 1, "Masters": 2, "PhD": 3}[ed_option]

# Occupation
st.write("His occupation:")
occu_option = st.selectbox(
    "Choose an option:", 
    ["Office Worker", "Business Owner", "Student", "Unemployment"], 
    key="occupation"
)
occu_type = {"Office Worker": -0.020, "Business Owner": 0.020, "Student": 0.004, "Unemployment": 0.001}[occu_option]

# Relationship Status
st.write("His relationship status:")
rel_option = st.selectbox(
    "Choose an option:", 
    ["In a Relationship", "Single", "Married", "Divorced"], 
    key="relationship_status"
)
rel_stat = {"In a Relationship": 0.028, "Single": -0.003, "Married": -0.012, "Divorced": -0.012}[rel_option]

# City Type
st.write("His living city type:")
city_option = st.selectbox(
    "Choose an option:", 
    ["Urban", "Rural", "Suburban"], 
    key="city_type"
)
city = {"Urban": -0.002, "Rural": 0.004, "Suburban": -0.002}[city_option]

# Empathy Score
empathy = st.number_input('Empathy score (1 to 100):')

# Anger Level
st.write("His anger level:")
anger_option = st.selectbox(
    "Choose an option:", 
    ["Low", "Moderate", "High"], 
    key="anger_level"
)
anger = {"Low": 1, "Moderate": 2, "High": 3}[anger_option]

# Boundary Respect
boundary = st.number_input('Respect for boundary (1 to 100):')

# Conflict Resolution
st.write("His conflict resolution status:")
conf_option = st.selectbox(
    "Choose an option:", 
    ["Passive", "Aggressive", "Rational", "Avoidant"], 
    key="conflict_resolution"
)
conf_res = {"Passive": 0.021, "Aggressive": -0.017, "Rational": -0.019, "Avoidant": 0.016}[conf_option]

# Online Comment Sentiment
online_cmnt = st.number_input('Rate his comment sentiment (-100 to +100). [Negative indicates negativity]:')

# Past Complaints
st.write("His past complaints of misbehavior:")
past_option = st.selectbox(
    "Choose an option:", 
    ["Workplace", "Social", "Multiple"], 
    key="past_complaints"
)
past = {"Workplace": 0.041, "Social": 0.097, "Multiple": -3.02}[past_option]

# Peer Rating
peer_rating = st.number_input('Friend and peer safety rating (0 to 5):')

# Social Good Participation
st.write("His participation in social good:")
soc_option = st.selectbox(
    "Choose an option:", 
    ["Yes", "No"], 
    key="social_good"
)
soc_good = {"Yes": 1, "No": 0}[soc_option]

# Alcohol Consumption
st.write("His alcohol consumption status:")
alcohol_option = st.selectbox(
    "Choose an option:", 
    ["Never", "Occasionally", "Frequently"], 
    key="alcohol_consumption"
)
alcohol = {"Never": 1, "Occasionally": 2, "Frequently": 3}[alcohol_option]

# History of Aggressive Behavior
st.write("His history of aggressive behavior:")
hist_option = st.selectbox(
    "Choose an option:", 
    ["Yes", "No"], 
    key="history_aggressive"
)
hist_aggressive = {"Yes": 1, "No": 0}[hist_option]

# Convert input to float array
input_data = [[age, ed_lvl, occu_type, rel_stat, city, empathy, anger, boundary, conf_res, online_cmnt, 
               past, peer_rating, soc_good, alcohol, hist_aggressive]]

# Prediction Button
if st.button('Predict'):
    try:
        prediction = model.predict(input_data)
        result = "Safe" if prediction[0] == 1 else "Unsafe"
        st.success(f"Prediction: {result}")
        st.write("**Interpretation:**")
        if prediction[0] == 1:
            st.write("Based on the given inputs, the model predicts that this individual is generally safe.")
        else:
            st.write("The model predicts that this individual may pose a threat. Further evaluation is recommended.")
    except Exception as e:
        st.error(f"Error: {e}")

# SHAP Interpretation Button
if st.button('XAI Interpretation'):
    try:
        # Convert input to DataFrame for SHAP
        input_df = pd.DataFrame(input_data, columns=['age', 'ed_lvl', 'occu_type', 'rel_stat', 'city', 
                                                     'empathy', 'anger', 'boundary', 'conf_res', 'online_cmnt', 
                                                     'past', 'peer_rating', 'soc_good', 'alcohol', 'hist_aggressive'])

        # Get SHAP values
        shap_values = explainer(input_df)

        # Feature Importance Plot (Summary Plot)
        st.subheader("Feature Importance (Summary Plot)")
        st.write("This plot shows which features had the greatest influence on the model's prediction. Features further from zero have a stronger impact. **Red** means the feature increased the risk, while **blue** means it reduced the risk.")
        plt.figure(figsize=(10, 5))
        shap.summary_plot(shap_values, input_df)
        st.pyplot(plt)

        # Bar plot
        st.subheader("Feature Impact (Bar Chart)")
        st.write("This bar chart ranks the features by their importance in making the prediction. Features at the top had the most influence on determining whether the individual is 'Safe' or 'Unsafe'.")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values.values, input_df, plot_type="bar")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error in SHAP interpretation: {e}")


