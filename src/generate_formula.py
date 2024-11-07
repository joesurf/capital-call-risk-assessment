import pandas as pd
import random
from faker import Faker

fake = Faker()

def generate_boolean():
    return random.choice([True, False])

def generate_date_of_birth():
    return fake.date_of_birth(minimum_age=20, maximum_age=70).strftime("%d.%m.%Y")

def generate_id():
    return random.randint(1000, 9999)

def generate_days():
    return random.randint(0, 182)

def generate_percent():
    return random.randint(-10, 10)

def generate_other_source(category):
    if category == "wealth":
        return random.choice(["IP ownership", "Crypto holdings", "Royalty earnings"])
    elif category == "funds":
        return random.choice(["Pension", "Grant", "Gift"])

def calculate_default_risk(row):
    stable_sources_wealth = row['Source of wealth salary'] + row['Source of wealth rental income'] + row['Source of wealth business revenue']
    high_risk_sources_wealth = row['Source of wealth investment gain'] + row['Source of wealth inheritance']
    
    stable_sources_funds = row['Source of funds salary'] + row['Source of funds rental income'] + row['Source of funds business revenue']
    high_risk_sources_funds = row['Source of funds investment gain'] + row['Source of funds inheritance']
    
    dob = row['Date of birth']
    year_of_birth = int(dob.split('.')[2])
    age = 2024 - year_of_birth
    age_risk = 0.05 if age < 30 else 0

    stable_sources = stable_sources_wealth + stable_sources_funds
    high_risk_sources = high_risk_sources_wealth + high_risk_sources_funds
    
    risk_score = (high_risk_sources - stable_sources) + age_risk
    
    if risk_score > 1:
        return True
    else:
        return False

data_100_realistic = {
    "Id": [generate_id() for _ in range(100)],
    "AI declaration": [generate_boolean() for _ in range(100)],
    "Date of birth": [generate_date_of_birth() for _ in range(100)],
    "Source of wealth salary": [generate_boolean() for _ in range(100)],
    "Source of wealth investment gain": [generate_boolean() for _ in range(100)],
    "Source of wealth inheritance": [generate_boolean() for _ in range(100)],
    "Source of wealth rental income": [generate_boolean() for _ in range(100)],
    "Source of wealth business revenue": [generate_boolean() for _ in range(100)],
    "Source of wealth other": [generate_other_source("wealth") for _ in range(100)],
    "Source of funds salary": [generate_boolean() for _ in range(100)],
    "Source of funds investment gain": [generate_boolean() for _ in range(100)],
    "Source of funds inheritance": [generate_boolean() for _ in range(100)],
    "Source of funds rental income": [generate_boolean() for _ in range(100)],
    "Source of funds business revenue": [generate_boolean() for _ in range(100)],
    "Source of funds other": [generate_other_source("funds") for _ in range(100)],
    "Days to capital call payment": [generate_days() for _ in range(100)],
    "Source of capital call payment": [generate_other_source("funds") for _ in range(100)],
    "GDP growth rate": [f"{generate_percent()}%" for _ in range(100)],
    "Interest rate": [f"{generate_percent()}%" for _ in range(100)],
}

df_realistic = pd.DataFrame(data_100_realistic)

df_realistic['default'] = df_realistic.apply(calculate_default_risk, axis=1)

synthetic_formula = "synthetic_formula.csv"
df_realistic.to_csv(synthetic_formula, index=False)
