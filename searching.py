import requests

def search_pubmed(query):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pmc",
        "term": query,
        "retmax": 1000  # Adjust the number of papers to retrieve
    }
    response = requests.get(base_url, params=params)
    return response.text

def save_pmids_to_file(pmids, filename):
    with open(filename, 'w') as file:
        for pmid in pmids:
            file.write(f"{pmid}\n")

# Example search query
query = "cardio"

# Fetch papers metadata from PubMed
search_results = search_pubmed(query)

# Extract PMIDs from the search results
pmids = search_results.split('<Id>')[1:]  # Split by <Id> tag and remove the first empty item

# Extract only the PMIDs from the extracted strings
pmids = [pmid.split('</Id>')[0] for pmid in pmids]

# Save retrieved PMIDs to a text file
save_pmids_to_file(pmids, 'pmids.txt')

print("PubMed IDs saved to pmids.txt")
