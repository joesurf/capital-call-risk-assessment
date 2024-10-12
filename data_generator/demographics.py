import pandas as pd
import numpy as np
import random
from faker import Faker

fake = Faker()

def generate_boolean():
    return random.choice([True, False])

def generate_date_of_birth():
    return fake.date_of_birth(minimum_age=20, maximum_age=70).strftime("%d.%m.%Y")

def generate_occupation(data_size: int=100) -> pd.DataFrame:
    occupation_df = pd.read_csv('data_reference/emp_res_detailed_occ.csv')
    latest_occupation_df = occupation_df[occupation_df['Year'] == 2023]
    
    occupation_gen = np.random.choice(
        latest_occupation_df['Detailed_Occupation'], 
        size=data_size, 
        p=latest_occupation_df['Emp_Res'] / latest_occupation_df['Emp_Res'].sum()
    )
    occupation_df = pd.DataFrame(occupation_gen, columns=['Occupation'])

    return occupation_df


def generate_age(data_size: int=100) -> pd.DataFrame:
    age_df = pd.read_csv('data_reference/Singapore Residents Age Sex.csv')
    
    age_df = age_df.rename(columns={'2024 ': '2024', 'Data Series': 'Age'})
    age_df['Age'] = age_df['Age'].apply(lambda x: int(x.split()[0]))

    total_population_2024 = age_df['2024'].sum()
    age_df['2024_percentage'] = (age_df['2024'] / total_population_2024) * 100

    new_ages = np.random.choice(
        age_df['Age'], 
        size=data_size, 
        p=age_df['2024'] / age_df['2024'].sum()
    )
    new_df = pd.DataFrame(new_ages, columns=['Age'])
    
    return new_df