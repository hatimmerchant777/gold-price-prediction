import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
import joblib

class GoldPricePredictor:
    def __init__(self):
        self.df = None
        self.models = {
            'linear': LinearRegression(),
            'logistic': LogisticRegression(max_iter=1000),
            'decision_tree': DecisionTreeRegressor(),
            'svm': SVR(kernel='rbf'),
            'naive_bayes': GaussianNB()
        }
        self.scaler = StandardScaler()
        self.discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        self.price_threshold = None
        
    def prepare_data(self):
        self.df = pd.read_csv('Gold Price (2013-2023).csv')
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Day'] = self.df['Date'].dt.day
        self.df['Price'] = self.df['Price'].str.replace(',', '').astype(float) * 83 * (10/31.1)
        self.price_threshold = self.df['Price'].median()
        features = ['Year', 'Month', 'Day']
        X = self.df[features]
        y = self.df['Price']
        X_scaled = self.scaler.fit_transform(X)
        y_discrete = self.discretizer.fit_transform(y.values.reshape(-1, 1))
        return X_scaled, y, y_discrete
        
    def train_models(self):
        X, y, y_discrete = self.prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        _, _, y_train_discrete, _ = train_test_split(X, y_discrete, test_size=0.2, random_state=42)
        results = {}
        for name, model in self.models.items():
            if name == 'logistic':
                y_binary = (y_train > self.price_threshold).astype(int)
                model.fit(X_train, y_binary)
            elif name == 'naive_bayes':
                model.fit(X_train, y_train_discrete.ravel())
            else:
                model.fit(X_train, y_train)
            results[name] = model
        return results
    
    def predict(self, year, month, day, model_name='linear'):
        input_data = np.array([[year, month, day]])
        input_scaled = self.scaler.transform(input_data)
        
        if model_name == 'naive_bayes':
            discrete_pred = self.models[model_name].predict(input_scaled)
            bin_means = [self.df['Price'].quantile(q) for q in np.linspace(0, 1, 6)[1:]]
            base_prediction = bin_means[int(discrete_pred[0])]
            years_ahead = year - 2023
            growth_factor = 1.0 + (years_ahead * 0.28)
            prediction = base_prediction * growth_factor * 1.55
            prediction = np.clip(prediction, 85000, 165000)
            
        elif model_name == 'logistic':
            pred_class = self.models[model_name].predict(input_scaled)[0]
            if pred_class == 1:
                prediction = float(self.df['Price'].mean() * 2.45)
            else:
                prediction = float(self.df['Price'].mean() * 2.25)
        else:
            base_prediction = self.models[model_name].predict(input_scaled)
            if isinstance(base_prediction, np.ndarray):
                prediction = base_prediction[0]
            else:
                prediction = base_prediction
            years_ahead = year - 2023
            growth_factor = 1.0 + (years_ahead * 0.28)
            prediction = prediction * growth_factor * 1.15
            prediction = np.clip(prediction, 90000, 180000)
        
        return float(prediction)