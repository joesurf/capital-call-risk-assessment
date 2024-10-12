import random
import pandas as pd


def generate_source_of_wealth(data_size: int=100) -> pd.DataFrame:
    """
        Assumptions:
        • Source of wealth doesn't have to add up to 100% since an individual can have multiple sources
        • Every individual should have at least one source of wealth

        others: IP ownership, crypto holdings, collectibles, etc.
    """
    wealth_source_distribution = {
        'wealth - salary': 0.35,
        'wealth - investment gain': 0.45,
        'wealth - rental income': 0.15,
        'wealth - business revenue': 0.1,
        'wealth - inheritance': 0.075,
        'wealth - others': 0.075
    }
    random_sorted_sources = [
        source for sublist in [
            [source] * int(perc * data_size) for source, perc in wealth_source_distribution.items()
        ] for source in sublist
    ]
    random.shuffle(random_sorted_sources)

    result = []
    for i in range(data_size):
        # Ensure each list has at least one item
        if not random_sorted_sources:
            break
        # Randomly decide the number of items in each sublist
        num_items = random.randint(1, max(1, len(random_sorted_sources) // (data_size - i)))
        sublist = []
        while len(sublist) < num_items and random_sorted_sources:
            source = random_sorted_sources.pop(0)
            if source not in sublist:
                sublist.append(source)
        result.append(sublist)
    
    # Convert the list of lists into a DataFrame with True/False values
    df_wealth_sources = pd.DataFrame([
        {source: source in sublist for source in wealth_source_distribution.keys()}
        for sublist in result
    ])
        
    return df_wealth_sources
