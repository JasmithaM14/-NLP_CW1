import xml.etree.ElementTree as ET
import pandas as pd

# Function to parse XML and extract article IDs and labels
def parse_ground_truth(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    articles = []
    for article in root.findall('article'):
        article_id = article.get('id')
        hyperpartisan = article.get('hyperpartisan')
        articles.append({'article_id': article_id, 'label': hyperpartisan})
    
    return pd.DataFrame(articles)

# Path to the XML file
xml_file = r'C:\NLP CW1\Hyperpartisian_News_Detection\pan-code\semeval19\input_directory\ground-truth-validation-bypublisher\Splits\split_1.xml'

# Parse the XML and create a DataFrame
df = parse_ground_truth(xml_file)

# Convert hyperpartisan labels from 'true'/'false' to 1/0
df['label'] = df['label'].map({'true': 1, 'false': 0})

# Save DataFrame to CSV
output_csv = 'ground-truth-training-bypublisher.csv'
df.to_csv(output_csv, index=False)

print(f"Ground truth labels have been saved to {output_csv}")