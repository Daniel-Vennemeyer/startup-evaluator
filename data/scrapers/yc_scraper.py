import requests
from bs4 import BeautifulSoup
import time
import json

# Base URLs
base_url = "https://www.ycombinator.com"

# Headers for requests to simulate a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Helper function to fetch and parse HTML
def get_soup(url):
    response = requests.get(url, headers=headers)
    return BeautifulSoup(response.text, "html.parser")

# Step 1: Scrape the list of companies
def get_company_links():
    file_path = 'scraper/yc_html.txt'

    with open(file_path, 'r') as file:
        html_content = file.read()

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract each company URL based on the <a> tag with class "_company_86jzd_338"
    company_links = soup.find_all("a", class_="_company_86jzd_338")

    # Create a list of URLs by appending the base URL
    company_links = [base_url + company["href"] for company in company_links]
    return company_links

# Step 2: Scrape individual company data
def scrape_company_data(company_url):
    soup = get_soup(company_url)
    
    # Extract the JSON from the `data-page` attribute
    data_div = soup.find("div", {"data-page": True})
    if not data_div:
        return None  # Skip if no data-page attribute
    
    # Parse JSON
    data = json.loads(data_div["data-page"])
    company_info = data.get("props", {}).get("company", {})
    
    # Extract desired fields
    name = company_info.get("name")
    one_liner = company_info.get("one_liner", "")
    description = company_info.get("long_description", "")
    
    
    return {
        "name": name,
        "description": description,
        "one_liner": one_liner,
    }


# Main function to scrape all company data
def scrape_all_companies():
    company_links = get_company_links()
    all_data = []
    
    for link in company_links:
        print(f"Scraping: {link}")
        company_data = scrape_company_data(link)
        all_data.append(company_data)
        
        # Respectful scraping: delay between requests
        time.sleep(1)
    
    return all_data

# Execute scraper and save results
data = scrape_all_companies()

# Save data as JSON for easy access
with open("yc_companies.json", "w") as f:
    json.dump(data, f, indent=4)
    
print("Scraping completed. Data saved to yc_companies.json")