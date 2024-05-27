import numpy as np
import polars as pl
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split


class HouseRocketModel:
    def __init__(self, dataframe):
        self.df = dataframe.to_pandas()
        self.model = None
        self.renovation_model = None
        self.features = [
            'num_bedrooms', 'num_bathrooms', 'living_area_sqft', 'num_floors', 'is_waterfront', 'view_rating',
            'condition', 'grade', 'above_grade_sqft', 'basement_sqft', 'year_built', 'year_renovated',
            'latitude', 'longitude', 'living_area_sqft15', 'lot_area_sqft15', 'years_since_renovation', 
            'sqft_per_bedroom', 'price_per_sqft', 'latitude_rounded', 'longitude_rounded', 
            'price_per_bedroom', 'price_per_bathroom'
        ]
        self.train_models()

    def clean_data(self, X):
        # Handle infinite values and fill NaNs
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        # Ensure the data is within the float32 range
        X = X.astype(np.float32)
        return X
    
    def train_models(self):
        X = self.clean_data(self.df[self.features].copy())
        y = self.df['listing_price'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the models
        rf = RandomForestRegressor(random_state=42)
        gbr = GradientBoostingRegressor(random_state=42)
        lr = LinearRegression()
        self.model = VotingRegressor(estimators=[('rf', rf), ('gbr', gbr), ('lr', lr)])
        
        # Cross-validation
        scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)
        print(f'Cross-validated RMSE: {rmse_scores.mean()}')

        self.model.fit(X_train, y_train)

        # Make predictions and evaluate the model
        predictions = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f'Main Model RMSE: {rmse}')
        
        # Train the renovation impact model
        renovation_data = self.df[self.df['condition'].isin([1, 3])].copy()
        renovation_data['price_increase'] = renovation_data['listing_price'] - self.model.predict(self.clean_data(renovation_data[self.features].copy()))

        X_renovation = self.clean_data(renovation_data[self.features].copy())
        y_renovation = renovation_data['price_increase']
        
        X_renovation_train, X_renovation_test, y_renovation_train, y_renovation_test = train_test_split(X_renovation, y_renovation, test_size=0.2, random_state=42)
        
        self.renovation_model = VotingRegressor(estimators=[('rf', rf), ('gbr', gbr), ('lr', lr)])
        
        renovation_scores = cross_val_score(self.renovation_model, X_renovation_train, y_renovation_train, cv=5, scoring='neg_mean_squared_error')
        renovation_rmse_scores = np.sqrt(-renovation_scores)
        print(f'Renovation Model Cross-validated RMSE: {renovation_rmse_scores.mean()}')

        self.renovation_model.fit(X_renovation_train, y_renovation_train)
        
        renovation_predictions = self.renovation_model.predict(X_renovation_test)
        renovation_rmse = np.sqrt(mean_squared_error(y_renovation_test, renovation_predictions))
        print(f'Renovation Model RMSE: {renovation_rmse}')

    def predict(self, new_data):
        new_data = self.clean_data(new_data.copy())
        predictions = self.model.predict(new_data[self.features])
        return predictions

    def predict_renovation_increase(self, new_data):
        new_data = self.clean_data(new_data.copy())
        predictions = self.renovation_model.predict(new_data[self.features])
        return predictions

    def predict_sale_price(self, new_data):
        new_data = self.clean_data(new_data.to_pandas().copy())

        # Predict the sale prices
        predictions = self.model.predict(new_data[self.features])
        new_data['predicted_price'] = predictions
        new_data['renovation_increase'] = 0

        # Identify renovation candidates
        renovation_candidates = new_data[new_data['condition'].isin([1, 3])]
        renovation_increase = self.predict_renovation_increase(renovation_candidates)
        
        # Cast renovation increase to int32
        renovation_increase = renovation_increase.astype(np.int32)

        # Update the renovation increase for candidates
        new_data.loc[new_data['condition'].isin([1, 3]), 'renovation_increase'] = renovation_increase

        # Add additional columns
        new_data['needs_renovation'] = new_data['condition'].apply(lambda x: x in [1, 3])
        new_data['potential_sale_no_renovation'] = new_data['predicted_price']
        new_data['potential_sale_with_renovation'] = new_data['predicted_price'] + new_data['renovation_increase']

        # Select the final columns to return
        final_data = new_data[['property_id', 'listing_price', 'potential_sale_no_renovation', 'needs_renovation', 'potential_sale_with_renovation']]

        return pl.from_pandas(final_data).with_columns(
            pl.col('property_id').cast(pl.Int64),
            pl.col('listing_price').cast(pl.Int64),
            pl.col('potential_sale_no_renovation').cast(pl.Int64),
            pl.col('potential_sale_with_renovation').cast(pl.Int64)
        )
