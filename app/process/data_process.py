
import polars as pl
from scipy.stats import chi2_contingency


class DataProcess:
    def __init__(self, path: str):
        self.path = path
        self.df = pl.read_csv(path).rename({
            'id': 'property_id',
            'date': 'transaction_date',
            'price': 'listing_price',
            'bedrooms': 'num_bedrooms',
            'bathrooms': 'num_bathrooms',
            'sqft_living': 'living_area_sqft',
            'sqft_lot': 'lot_area_sqft',
            'floors': 'num_floors',
            'waterfront': 'is_waterfront',
            'view': 'view_rating',
            'condition': 'condition',
            'grade': 'grade',
            'sqft_above': 'above_grade_sqft',
            'sqft_basement': 'basement_sqft',
            'yr_built': 'year_built',
            'yr_renovated': 'year_renovated',
            'zipcode': 'zip_code',
            'lat': 'latitude',
            'long': 'longitude',
            'sqft_living15': 'living_area_sqft15',
            'sqft_lot15': 'lot_area_sqft15'
        })
        self.clean_data()

    def clean_data(self):
        self.df = self.df.with_columns(
            pl.col('transaction_date').str.strptime(pl.Datetime, "%Y%m%dT%H%M%S", strict=False)
        )

        self.df = self.df.fill_null(strategy="mean")
        
        self.df = self.df.with_columns([
            pl.col('transaction_date').dt.timestamp(),
            pl.col('listing_price').cast(pl.Int64),
            pl.col('num_bedrooms').cast(pl.Int64),
            pl.col('latitude').cast(pl.Float64),
            pl.col('longitude').cast(pl.Float64),
            pl.col('living_area_sqft15').cast(pl.Int64),
            pl.col('lot_area_sqft15').cast(pl.Int64)
        ])

        # Features for model
        self.df = self.df.with_columns([
            (pl.col('year_built') - pl.col('year_renovated')).alias('years_since_renovation'),
            (pl.col('living_area_sqft') / pl.col('num_bedrooms')).alias('sqft_per_bedroom'),
            (pl.col('listing_price') / pl.col('living_area_sqft')).alias('price_per_sqft'),
            (pl.col('latitude').round(2)).alias('latitude_rounded'),
            (pl.col('longitude').round(2)).alias('longitude_rounded'),
            (pl.col('listing_price') / pl.col('num_bedrooms')).alias('price_per_bedroom'),
            (pl.col('listing_price') / pl.col('num_bathrooms')).alias('price_per_bathroom'),
        ])

    def get_data(self) -> pl.DataFrame:
        return self.df
    
    def correlation_data(self) -> pl.DataFrame:
        return self.df.corr()
    
    def avg_time_to_renovate(self) -> float:
        renovation_times = self.df.filter(pl.col('year_renovated') > 0).with_columns(
            (pl.col('year_renovated') - pl.col('year_built')).alias('time_to_renovate')
        ).select('time_to_renovate')
        return renovation_times.mean()
    
    def price_difference_renovation(self) -> pl.DataFrame:
        renovated = self.df.filter(pl.col('year_renovated') > 0)
        non_renovated = self.df.filter(pl.col('year_renovated') == 0)
        return pl.DataFrame({
            'renovated_avg_price': [renovated['listing_price'].mean()],
            'non_renovated_avg_price': [non_renovated['listing_price'].mean()]
        })

    def renovation_at_sale_or_purchase(self) -> pl.DataFrame:
        self.df = self.df.with_columns(
            (pl.col('year_renovated') == pl.col('transaction_date').cast(pl.Datetime).dt.year()).alias('renovated_at_sale')
        )
        return self.df.groupby('renovated_at_sale').agg(pl.count())
    
    def test_independence_condition_year_renovated(self):
        contingency_table = self.df.select(['condition', 'year_renovated']).to_pandas().pivot_table(index='condition', columns='year_renovated', aggfunc='size', fill_value=0)
        
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        
        return {'chi2': chi2, 'p_value': p, 'degrees_of_freedom': dof, 'expected_frequencies': ex}

    def recommend_renovation(self):
        total_count = self.df.groupby('condition').agg(pl.count('condition').alias('total'))
        renovated_count = self.df.filter(pl.col('year_renovated') > 0).groupby('condition').agg(pl.count('condition').alias('renovated'))

        recommendation = total_count.join(renovated_count, on='condition', how='left').fill_null(0).with_columns(
            (pl.col('renovated') / pl.col('total')).alias('renovation_rate')
        ).select(['condition', 'renovation_rate'])

        return recommendation

    def potential_valuation(self, conditions: list[int]) -> pl.DataFrame:
        results = []
        for condition in conditions:
            filtered_data = self.df.filter(pl.col('condition') == condition)
            renovated = filtered_data.filter(pl.col('year_renovated') > 0)
            non_renovated = filtered_data.filter(pl.col('year_renovated') == 0)
            renovated_avg_price = renovated['listing_price'].mean()
            non_renovated_avg_price = non_renovated['listing_price'].mean()
            price_difference = renovated_avg_price - non_renovated_avg_price
            num_renovated = renovated.height
            num_non_renovated = non_renovated.height
            renovation_proportion = num_renovated / (num_renovated + num_non_renovated) if (num_renovated + num_non_renovated) > 0 else 0
            results.append((condition, renovated_avg_price, non_renovated_avg_price, price_difference, num_renovated, num_non_renovated, renovation_proportion))
        
        return pl.DataFrame(
            results,
            schema=["condition", "renovated_avg_price", "non_renovated_avg_price", "price_difference", "num_renovated", "num_non_renovated", "renovation_proportion"]
        )

    def check_nulls(self) -> pl.DataFrame:
        return self.df.null_count()