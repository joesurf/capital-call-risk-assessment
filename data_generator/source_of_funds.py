import random
import pandas as pd


def generate_source_of_funds(data_size: int=100) -> pd.DataFrame:
    """
        Assumptions:
        • Source of funds doesn't have to add up to 100% since an individual can have multiple sources
        • Every individual should have at least one source of fund
        • Not possible to find distribution for funds for capital calls - use wealth as baseline and adjust for context

        others: dividends, pension, royalties, etc.
    """
    fund_source_distribution = {
        'fund - salary': 0.5,
        'fund - investment gain': 0.1,
        'fund - rental income': 0.1,
        'fund - business revenue': 0.1,
        'fund - inheritance': 0.1,
        'fund - others': 0.1
    }
    random_sorted_sources = [
        source for sublist in [
            [source] * int(perc * data_size) for source, perc in fund_source_distribution.items()
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
    df_fund_sources = pd.DataFrame([
        {source: source in sublist for source in fund_source_distribution.keys()}
        for sublist in result
    ])
        
    return df_fund_sources
