import pandas as pd

def process_energy_data(input_file, output_file):
    df = pd.read_csv(
        input_file,
        sep=';',
        parse_dates={'Datetime': ['Date', 'Time']},
        infer_datetime_format=True,
        low_memory=False,
        na_values=['?'],
        index_col='Datetime'
    )

    aggregation_rules = {
        'Global_active_power': 'mean',
        'Global_reactive_power': 'mean',
        'Voltage': 'mean',
        'Global_intensity': 'max',
        'Sub_metering_1': 'mean',
        'Sub_metering_2': 'mean',
        'Sub_metering_3': 'mean'
    }

    df_hourly = df.resample('H').agg(aggregation_rules)
    df_hourly = df_hourly.interpolate(method='linear')
    df_hourly.to_csv(output_file)

if __name__ == "__main__":
    process_energy_data('household_power_consumption.txt', 'household_hourly_dataset.csv')