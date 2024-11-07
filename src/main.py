import pandas as pd

from data_generator.source_of_funds import generate_source_of_funds
from data_generator.source_of_wealth import generate_source_of_wealth
from data_generator.demographics import generate_age, generate_occupation
from data_generator.ai_declaration import generate_ai_declaration


if __name__ == '__main__':
    df_speed_of_payment = pd.read_csv('data_reference/synthetic_behavioural.csv')

    df_source_of_wealth = generate_source_of_wealth(data_size=1000)
    df_source_of_funds = generate_source_of_funds(data_size=1000)
    df_age = generate_age(data_size=1000)
    df_occupation = generate_occupation(data_size=1000)
    df_ai_declaration = generate_ai_declaration(data_size=1000)

    df_combined = pd.concat([
        df_source_of_wealth, 
        df_source_of_funds, 
        df_age, df_occupation, 
        df_speed_of_payment,
    ], axis=1)
    df_combined.to_csv('synthetic_distribution.csv', index=False)

    synthetic_describe = df_combined.describe(include='all')
    synthetic_describe.to_csv('synthetic_describe.csv', index=False)

