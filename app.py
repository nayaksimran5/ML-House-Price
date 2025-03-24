import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Load the model
def load_model(model_dir: str = "."):
    model_path = os.path.join(model_dir, "xg_boost_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)

# Load the data to get actual ranges
data = pd.read_csv("data/test_df_v1.csv")

# Default scaled values (unchanged from original)
default_scaled_values = {
    'NeighborhoodQuality': -0.17038732930250838,
    'OverallQual': -0.5,
    'TotalBsmtSF': -0.20384615384615384,
    'BsmtFinSF1': 0.15691489361702127,
    'GarageSize': -0.2689075630252101,
    'TotalBath': -0.6666666666666666,
    'CentralAir_Y': 1.0,
    'OverallCond': 1.0,
    'LotArea': 0.5387131952017448,
    '1stFlrSF': -0.35952848722986247,
    'YearsSinceRemodel': 0.825,
    'GarageCars': -1.0,
    'BsmtUnfSF': -0.328719723183391,
    'Age': 0.3125,
    'YearRemodAdd': -0.7560975609756098,
    'YearBuilt': -0.25,
    'GarageArea': 0.9689922480620154,
    'MSZoning_RM': 0.0,
    'GrLivArea': -0.8881524440762221,
    'TotalOutdoorSF': 0.43317972350230416,
    'LotFrontage': 0.6111111111111112,
    '2ndFlrSF': 0.0,
    'MoSold': 0.0,
    'GarageFinish_Unf': 1.0,
    'TotalPorchSF': 0.47794117647058826,
    'OpenPorchSF': -0.3888888888888889,
    'SaleCondition_Normal': 1.0,
    'GarageYrBlt': -0.41975308641975306,
    'NeighborhoodQuality': -0.17
}

# Define actual min-max ranges from the loaded data
feature_ranges = {
    'OverallQual': (data['OverallQual'].min(), data['OverallQual'].max()),
    'GrLivArea': (data['GrLivArea'].min(), data['GrLivArea'].max()),
    'GarageCars': (data['GarageCars'].min(), data['GarageCars'].max()),
    'TotalBsmtSF': (data['TotalBsmtSF'].min(), data['TotalBsmtSF'].max()),
    'TotalBath': (data['TotalBath'].min(), data['TotalBath'].max()),
    'YearBuilt': (data['YearBuilt'].min(), data['YearBuilt'].max()),
    'Qual_LivArea': (data['Qual_LivArea'].min(), data['Qual_LivArea'].max()),
    'TotalFlrSF': (data['TotalFlrSF'].min(), data['TotalFlrSF'].max())
}

def scale_to_original(ui_value, feature_name):
    """Convert 0-1 UI value to original scale based on feature's min-max range"""
    min_val, max_val = feature_ranges[feature_name]
    return min_val + (ui_value * (max_val - min_val))

def predict_price(OverallQual, GrLivArea, GarageCars, TotalBsmtSF, 
                 TotalBath, YearBuilt, Qual_LivArea, TotalFlrSF):
    # Load model and start with default scaled values
    model = load_model()
    scaled_dict = default_scaled_values.copy()

    # Convert UI 0-1 values back to original scale using actual ranges
    scaled_dict.update({
        'OverallQual': scale_to_original(OverallQual, 'OverallQual'),
        'GrLivArea': scale_to_original(GrLivArea, 'GrLivArea'),
        'GarageCars': scale_to_original(GarageCars, 'GarageCars'),
        'TotalBsmtSF': scale_to_original(TotalBsmtSF, 'TotalBsmtSF'),
        'TotalBath': scale_to_original(TotalBath, 'TotalBath'),
        'YearBuilt': scale_to_original(YearBuilt, 'YearBuilt'),
        'Qual_LivArea': scale_to_original(Qual_LivArea, 'Qual_LivArea'),
        'TotalFlrSF': scale_to_original(TotalFlrSF, 'TotalFlrSF')
    })
    
    # Create input DataFrame directly from scaled values in model's expected order
    input_data = pd.DataFrame([scaled_dict], columns=model.feature_names_in_)
    
    # Make prediction
    prediction = np.expm1(model.predict(input_data))
    return f"Predicted Sale Price: ${prediction[0]:,.2f}"

# Convert default values to 0-1 scale for UI display
def to_ui_scale(value, feature_name):
    min_val, max_val = feature_ranges[feature_name]
    return (value - min_val) / (max_val - min_val)

# Create Gradio interface with sliders scaled to 0-1 for display
iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Slider(0, 1, value=to_ui_scale(-0.5, 'OverallQual'), step=0.01, label="Overall Quality (1-10 raw)"),
        gr.Slider(0, 1, value=to_ui_scale(-0.8881524440762221, 'GrLivArea'), step=0.01, label="Living Area (334-5642 sqft raw)"),
        gr.Slider(0, 1, value=to_ui_scale(-1.0, 'GarageCars'), step=0.01, label="Garage Cars (0-4 raw)"),
        gr.Slider(0, 1, value=to_ui_scale(-0.20384615384615384, 'TotalBsmtSF'), step=0.01, label="Basement Area (0-6110 sqft raw)"),
        gr.Slider(0, 1, value=to_ui_scale(-0.6666666666666666, 'TotalBath'), step=0.01, label="Bathrooms (1-6 raw)"),
        gr.Slider(0, 1, value=to_ui_scale(-0.25, 'YearBuilt'), step=0.01, label="Year Built (1872-2010 raw)"),
        gr.Slider(0, 1, value=to_ui_scale(-0.17, 'Qual_LivArea'), step=0.01, label="Living Area Quality (334-56420 raw)"),
        gr.Slider(0, 1, value=to_ui_scale(-0.17, 'TotalFlrSF'), step=0.01, label="Total Floor SF (334-11752 raw)")
    ],
    outputs="text",
    title="House Price Prediction",
    description="Adjust the 8 key features using scaled values (0 to 1 for display). Values are transformed to original scale for prediction."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8080, share = True)