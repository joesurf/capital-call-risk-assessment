import pandas as pd



def read_files():
    age_risk_profile = pd.read_csv('data_label_criteria/Age Risk Profile.csv')
    net_worth_risk_profile = pd.read_csv('data_label_criteria/Net Worth Profile.csv')
    occupation_risk_profile = pd.read_csv('data_label_criteria/Occupation Risk Profile.csv')
    source_of_wealth_risk_profile = pd.read_csv('data_label_criteria/Source of Wealth Risk Profile.csv')
    income_risk_profile = pd.read_csv('data_label_criteria/Income Risk Profile.csv')






if __name__ == "__main__":



    read_files()