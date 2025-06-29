import pandas as pd
import numpy as np

# Original columns from your dataset
columns = [
    'Crime Description', 'Weapon Used', 'Crime Severity', "Victim's Injuries",
    'Evidence Availability', 'Witness Statements', 'Citation', 'Crime Type',
    "Criminal's Previous Record", 'Criminal Mental Health', 'Evidence Strength',
    "Criminal's Bail Status", 'Sentencing Outcome'
]

# Define possible values for each category
crime_types = {
    'Theft': 'Section 382 - Punishment for Theft',
    'Murder': 'Section 302 - Punishment for Murder',
    'Fraud': 'Section 420 - Cheating and Fraud',
    'Robbery': 'Section 392 - Punishment for Robbery',
    'Kidnapping': 'Section 365 - Punishment for Kidnapping'
}
weapons = ['Knife', 'Handgun', 'None']
severities = {
    'Theft': 'Low',
    'Murder': 'High',
    'Fraud': 'Medium',
    'Robbery': 'High',
    'Kidnapping': 'High'
}
injuries = ['None', 'Moderate physical harm', 'Fatal injury', 'Psychological trauma']
evidence_types = ['Fingerprints', 'Digital records', 'Surveillance audio', 'DNA evidence']
witness_statements = ['Supported the victim', 'Contradicted alibi', 'Confirmed timeline']
previous_record = ['Yes', 'No']
mental_health = ['Stable', 'Unstable']
evidence_strength = ['Weak', 'Moderate', 'Strong']
bail_status = ['Granted', 'Denied']

# Helper function for generating fines based on crime type
def generate_fine(crime_type):
    fines = {
        'Theft': f'Up to Rs. {np.random.randint(10000, 100000)}',
        'Murder': f'Compensation to victim\'s family - Rs. {np.random.randint(500000, 1000000)}',
        'Fraud': f'Equal to the amount defrauded or more - Rs. {np.random.randint(100000, 500000)}',
        'Robbery': f'Up to Rs. {np.random.randint(50000, 200000)}',
        'Kidnapping': f'Up to Rs. {np.random.randint(100000, 300000)}'
    }
    return fines[crime_type]

# Helper function for logical sentencing
def generate_outcome(crime_type, strength):
    fine = generate_fine(crime_type)
    if strength == 'Strong':
        if crime_type == 'Murder':
            return np.random.choice(['Death Penalty', 'Life Imprisonment']) + f' + Fine: {fine}'
        elif crime_type == 'Robbery':
            return '10 to 14 years imprisonment + Fine: ' + fine
        elif crime_type == 'Fraud':
            return 'Up to 7 years imprisonment + Fine: ' + fine
        elif crime_type == 'Kidnapping':
            return 'Up to 7 years imprisonment + Fine: ' + fine
        else:
            return 'Up to 5 years imprisonment + Fine: ' + fine
    else:
        return np.random.choice(['Probation', f'Fine: {fine}', 'Community Service'])

# Helper function for crime descriptions
def generate_description(crime_type):
    descriptions = {
        'Theft': "The suspect unlawfully took valuables from the victim's property.",
        'Murder': "The suspect intentionally caused the death of the victim.",
        'Fraud': "The suspect engaged in deceitful activities for financial gain.",
        'Robbery': "The suspect forcefully took belongings from the victim using threats or violence.",
        'Kidnapping': "The suspect illegally confined or transported the victim against their will."
    }
    return descriptions[crime_type]

# Generate 2000 records (400 for each crime type)
data = []
for crime, citation in crime_types.items():
    for _ in range(400):
        strength = np.random.choice(evidence_strength)
        data.append([
            generate_description(crime),
            np.random.choice(weapons),
            severities[crime],
            np.random.choice(injuries),
            ', '.join(np.random.choice(evidence_types, size=2, replace=False)),
            np.random.choice(witness_statements),
            citation,
            crime,
            np.random.choice(previous_record),
            np.random.choice(mental_health),
            strength,
            np.random.choice(bail_status),
            generate_outcome(crime, strength)
        ])

# Create the DataFrame
final_df = pd.DataFrame(data, columns=columns)

# Save to CSV
final_df.to_csv('Pakistan_Crime_Dataset.csv', index=False)
print("Balanced dataset successfully created and saved as 'Pakistan_Crime_Dataset.csv'.")
