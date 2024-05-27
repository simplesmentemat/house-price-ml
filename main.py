import polars as pl

from app.model.model import HouseRocketModel
from app.process.data_process import DataProcess


def main():
    path = r"app/data/kc_house_data.csv"
    data = DataProcess(path)
        
    df = data.get_data()
    df = df.with_columns(pl.col("listing_price").sort(descending=True))
    
    # correlations = data.correlation_data()
    # print(correlations.write_csv("correlations.csv"))

    # avg_time = data.avg_time_to_renovate()
    # print(f"Average time to renovate: {avg_time} years")

    # price_diff = data.price_difference_renovation()
    # print(price_diff)

    # renovation_timing = data.renovation_at_sale_or_purchase()
    # print(renovation_timing)

    # year_renovated_condition_test = data.test_independence_condition_year_renovated()
    # print(year_renovated_condition_test)

    # recommendation = data.recommend_renovation()
    # print(recommendation)

    # var_1_3 = data.potential_valuation([1, 3])
    # print(var_1_3)

    # model = HouseRocketModel(df)
    # predictions = model.predict_sale_price(df)
    # print(predictions.write_csv("predictions.csv"))

    recomend = pl.read_csv("predictions.csv")

    recomend = recomend.with_columns(
        (pl.col('potential_sale_no_renovation') - pl.col('listing_price')).alias('potential_profit_no_renovation')
    )

    recomend = recomend.with_columns(
        (pl.col('potential_sale_with_renovation') - pl.col('listing_price')).alias('potential_profit_with_renovation')
    )

    recomend = recomend.select(["property_id", "listing_price", "potential_profit_no_renovation", "potential_profit_with_renovation"]).sort('potential_profit_with_renovation', descending=True)

    top_recommendation = recomend.head(10)

    print(top_recommendation)
if __name__ == "__main__":
    main()