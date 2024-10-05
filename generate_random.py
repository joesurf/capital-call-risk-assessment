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

def generate_other_source(category):
    if category == "wealth":
        return random.choice(["IP ownership", "Crypto holdings", "Royalty earnings"])
    elif category == "funds":
        return random.choice(["Pension", "Grant", "Gift"])

def generate_default():
    return random.choices([True, False], weights=[0.12, 0.88])[0]

data_100 = {
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
    "default": [generate_default() for _ in range(100)]
}

df_100 = pd.DataFrame(data_100)

synthetic_random = "synthetic_random.csv"
df_100.to_csv(synthetic_random, index=False)
