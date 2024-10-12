import random
import pandas as pd
import numpy as np


def generate_ai_declaration(data_size: int=100) -> pd.DataFrame:
    """
        Assumptions:
        • 

    """
    personal_income_distribution = {
        'income': 0.7,
        'real assets': 0.1,
        'financial assets': 0.2,
    }
    
    ai_declaration_list = np.random.choice(
        list(personal_income_distribution.keys()), 
        size=data_size, 
        p=[item[1] for item in personal_income_distribution.items()]
    )
    ai_declaration_df = pd.DataFrame(ai_declaration_list, columns=['ai_declaration'])

    return ai_declaration_df