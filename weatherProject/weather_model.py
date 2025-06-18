import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os
import sys
from datetime import datetime

def train_weather_model(csv_path):
    try:
        # Load data
        df = pd.read_csv(csv_path)
        print("\nSuccessfully loaded data. Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        
        # Since there's no date column, we'll create one
        print("\nCreating date index...")
        df['date'] = pd.date_range(start='2000-01-01', periods=len(df), freq='D')
        
        # Feature engineering
        print("Creating time-based features...")
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Prepare features and targets
        print("\nPreparing features and targets...")
        feature_columns = [
            'year', 'month', 'day', 'day_of_year',
            'MinTemp', 'MaxTemp', 'Humidity', 'Pressure'
        ]
        
        target_columns = {
            'temp': 'Temp',
            'min_temp': 'MinTemp',
            'max_temp': 'MaxTemp',
            'humidity': 'Humidity'
        }
        
        # Add wind speed if available
        if 'WindGustSpeed' in df.columns:
            feature_columns.append('WindGustSpeed')
            print(" - Added WindGustSpeed to features")
        
        print("Features:", feature_columns)
        print("Targets:", target_columns)
        
        # Train models for each target
        model_dir = os.path.join(os.path.dirname(__file__), 'ml_models')
        os.makedirs(model_dir, exist_ok=True)
        
        for model_name, target_col in target_columns.items():
            # Skip if target column doesn't exist
            if target_col not in df.columns:
                print(f"\nSkipping {model_name} - {target_col} column not found")
                continue
                
            # Prepare data
            X = df[feature_columns]
            y = df[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            print(f"\nTraining {model_name} model for {target_col}...")
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, pred)
            print(f" - MAE: {mae:.2f}")
            print(f" - Sample: Predicted {pred[0]:.2f} vs Actual {y_test.iloc[0]:.2f}")
            
            # Save model
            model_path = os.path.join(model_dir, f'{model_name}_model.pkl')
            joblib.dump(model, model_path)
            print(f" - Saved model to {model_path}")
        
        print("\nTraining completed successfully!")
        print(f"Models saved to: {model_dir}")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python weather_model.py <path_to_csv>")
        print("Example: python weather_model.py \"weather.csv\"")
    else:
        csv_path = sys.argv[1]
        print(f"\nStarting weather model training with: {csv_path}")
        train_weather_model(csv_path)