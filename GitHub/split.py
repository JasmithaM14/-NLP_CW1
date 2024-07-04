import xml.etree.ElementTree as ET

def get_article_id(article):
    # Assuming the article ID is an attribute or a sub-element
    # Adjust this function based on your XML structure
    return article.find('article_id').text

def split_xml(input_file, output_dir, articles_per_file):
    # Parse the input XML file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Create the root element for the new XML files
    articles_tag = root.tag
    article_tag = root[0].tag

    # Extract and sort articles by their IDs
    articles = sorted(root, key=get_article_id)

    total_articles = len(articles)
    num_files = (total_articles // articles_per_file) + (1 if total_articles % articles_per_file != 0 else 0)

    for i in range(num_files):
        # Create a new XML root
        new_root = ET.Element(articles_tag)

        # Add articles to the new XML root
        for article in articles[i * articles_per_file: (i + 1) * articles_per_file]:
            new_root.append(article)

        # Write the new XML file
        new_tree = ET.ElementTree(new_root)
        new_tree.write(f"{output_dir}/split_{i+1}.xml", encoding='utf-8', xml_declaration=True)

    print(f"Split into {num_files} files.")

# Example usage
input_file = r'C:\NLP CW1\Hyperpartisian_News_Detection\pan-code\semeval19\input_directory\articles-training-bypublisher\articles-training-bypublisher-20181122.xml'
output_dir = r'C:\NLP CW1\Hyperpartisian_News_Detection\pan-code\semeval19\input_directory\articles-training-bypublisher'
articles_per_file = 1500  # Adjust this number based on your memory capacity

split_xml(input_file, output_dir, articles_per_file)