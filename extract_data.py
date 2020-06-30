from functions import extract_data

# Variables

# Extract data
disease_name = 'PNEUMONIA'
database_path = '../MIMIC-III'
#-----------

alive_df, dead_df = extract_data(disease_name, database_path)

print(alive_df.head())
print('-'*60)
print(dead_df.head())