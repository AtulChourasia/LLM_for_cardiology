import os
import requests

def fetch_papers_from_pmids(pmids_file, output_dir):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    db = "pmc"
    retmode = "xml"  # You can change this to "text" for plain text format

    with open(pmids_file, 'r') as f:
        pmids = f.read().splitlines()

    for pmid in pmids:
        params = {
            "db": db,
            "id": pmid,
            "retmode": retmode
        }
        response = requests.get(base_url, params=params)
        content = response.text

        # Save paper to file
        output_file = os.path.join(output_dir, f"paper_{pmid}.xml")  # Change the extension if needed
        with open(output_file, 'w') as paper_file:
            paper_file.write(content)

        print(f"Paper {pmid} downloaded and saved as {output_file}")

# Example usage:
pmids_file = 'pmids.txt'  # Text file containing PubMed IDs
output_directory = 'papers'  # Directory where papers will be saved

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

fetch_papers_from_pmids(pmids_file, output_directory)
