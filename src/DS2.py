from src.zero_shot_eval import evaluate_domain_shift

# The translator for IIT Bhubaneswar
ds2_mapping = {
    'flyash': 'ash',
    'slag (GGBS)': 'slag',
    'SP': 'superplastic',
    'Coarse Agg.': 'coarseagg',
    'SAND (fine agg.)': 'fineagg',
    'AGE': 'age',
    'Compresive Strength': 'strength'
}

# Run the test
evaluate_domain_shift('Database_IIT_BBS_LAB_Concrete.csv', rename_map=ds2_mapping)