from src.zero_shot_eval import evaluate_domain_shift

# The translator for HPC Figshare
ds3_mapping = {
    'Fine\nAggregate': 'fineagg',
    'Coarse\nAggregate': 'coarseagg',
    'Fly ash': 'ash',
    'Cement': 'cement',
    'Slag': 'slag',
    'Water': 'water',
    'Superplasticizer': 'superplastic',
    'Age': 'age',
    'Compressive \nStrength': 'strength'
}

# Run the test
evaluate_domain_shift('Data Compressive Strength.csv', rename_map=ds3_mapping)