# Jose Valerio
# CAP5619 Fall 2026
# Github repository:
# https://github.com/valerio45a/Assignment-2---LLM-Document-Analysis.git
"""
AI Use Disclosure:
In completing this assignment, an AI language model was utilized to draft the
Python extraction pipeline, implement the SEC EDGAR API URL routing, and configure the
Named Entity Recognition (NER) prompt for the Gemini API. Output formatting and
API integration were independently verified to ensure adherence to assignment constraints.

Citation:
Google Gemini. Accessed 2026-11-04. Prompt: 'In this single-student assignment, you will
analyze SEC Form 8-K filings... extract new product releases and structure the data
in a tabular format.' Generated using Google Gemini.
"""
# Install if packages are missing:
# pip install google-genai pydantic
# pip install lxml

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import subprocess

# Gemini Imports - remove the '#' to use Gemini
# from google import genai
# from google.genai import types
# from pydantic import BaseModel, Field
# import getpass

# The SEC explicitly requires a User-Agent with contact info to prevent blocking.
headers = {'User-Agent': 'Student Name your_email@knights.ucf.edu'}
OLLAMA_MODEL = "llama3.2"


# Gemini Configuration - remove the '#' to use Gemini
# api_key_input = input("Enter your Gemini API key: ")
# client = genai.Client(api_key=api_key_input)

# class ProductRelease(BaseModel):
#     company_name: str
#     stock_name: str
#     filing_time: str
#     new_product: str = Field(description="Extracted Product Name (or None if not found)")
#     product_description: str = Field(description="Concise description strictly under 180 characters (or None)")

# SEC Data Retrieval
def get_cik(ticker):
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=headers)
    data = response.json()
    for key, value in data.items():
        if value['ticker'] == ticker:
            return str(value['cik_str']).zfill(10), value['title']
    return None, None


def get_8k_filings(cik, count=10):
    search_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K&count={count}&output=atom"
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.content, 'xml')

    filings = []
    for entry in soup.find_all('entry'):
        filing_time = entry.updated.text if entry.updated else "Unknown"
        link = entry.link['href'] if entry.link else None

        if link:
            text_link = link.replace('-index.htm', '.txt')
            filings.append({'time': filing_time, 'url': text_link})

    return filings


def get_8k_text(url):
    print("    (Downloading document...)")
    try:
        # 10-second timeout prevents the script from hanging if the SEC server stalls
        response = requests.get(url, headers=headers, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"    (Failed to download: {e})")
        return ""

    print("    (Parsing HTML...)")
    # Grab only the first 50,000 characters of raw HTML.
    # Stops BeautifulSoup from freezing on 50MB Base64 image encodings.
    raw_html = response.text[:50000]
    soup = BeautifulSoup(raw_html, 'lxml')

    print("    (Sending to LLM...)")
    return soup.get_text()[:5000]


# Ollama Extraction Logic
def extract_product_ollama(text, company_name, ticker, filing_time):
    prompt = f"""
    You are a Named Entity Recognition (NER) assistant.
    Review the following SEC 8-K filing text and extract ANY NEW PRODUCT RELEASES.

    Rules:
    1. Only extract ACTUAL NEW PRODUCTS. Ignore general business updates or financial results.
    2. The product description MUST be under 180 characters.

    Expected JSON format:
    {{
        "company_name": "{company_name}",
        "stock_name": "{ticker}",
        "filing_time": "{filing_time}",
        "new_product": "Product Name (or None if not found)",
        "product_description": "Description under 180 characters (or None)"
    }}

    Text to analyze:
    {text}
    """

    try:
        # Talk to Ollama's background server directly, avoiding Windows CLI limits
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json"  # Forces Ollama to strictly output valid JSON
            },
            timeout=120  # Give GPU up to 2 minutes to generate the answer
        )

        if response.status_code == 200:
            result_text = response.json().get("response", "{}")
            return json.loads(result_text)
        else:
            print(f"    (Ollama Error: {response.status_code} - Is the Ollama app running?)")
            return None

    except Exception as e:
        print(f"    (Error connecting to Ollama: {e})")
        return None


# Gemini Extraction Logic - Remove '#' to use
# def extract_product_gemini(text, company_name, ticker, filing_time):
#     prompt = f"""
#     You are a financial data extraction assistant performing Named Entity Recognition (NER).
#     Review the following SEC 8-K filing text and extract information about any NEW PRODUCT RELEASES.
#     Rules:
#     1. Only extract actual new products, not general business updates or financial results.
#     2. Populate the company name as '{company_name}', ticker as '{ticker}', and filing time as '{filing_time}'.
#
#     Text to analyze:
#     {text}
#     """
#     try:
#         response = client.models.generate_content(
#             model='gemini-3.8-flash',
#             contents=prompt,
#             config=types.GenerateContentConfig(
#                 response_mime_type="application/json",
#                 response_schema=ProductRelease,
#                 temperature=0.0
#             )
#         )
#         return response.parsed.model_dump()
#     except Exception as e:
#         print(f"Error querying Gemini: {e}")
#         return None


# Main Execution Pipeline
if __name__ == "__main__":
    tickers_to_check = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "INTC",
        "AMD", "NFLX", "DIS", "V", "JNJ", "WMT", "JPM", "PG", "MA", "HD", "UNH", "VZ"
    ]
    all_extracted_data = []

    for ticker in tickers_to_check:
        print(f"Processing {ticker}...")
        cik, company_title = get_cik(ticker)

        if not cik:
            print(f"Could not find CIK for {ticker}")
            continue

        filings = get_8k_filings(cik, count=5)

        for filing in filings:
            print(f"  Reading filing from {filing['time'][:10]}...")
            text = get_8k_text(filing['url'])

            # If the download failed, skip to the next document
            if not text:
                continue

            # Toggle LLM
            # Defaulting to Ollama. To use Gemini, add a '#' to the ollama line and remove the '#' from the gemini line.
            data = extract_product_ollama(text, company_title, ticker, filing['time'][:10])
            # data = extract_product_gemini(text, company_title, ticker, filing['time'][:10])

            if data and data.get('new_product') not in ["None", None, ""]:
                all_extracted_data.append(data)

            time.sleep(0.5)

            # Format and Export
    if all_extracted_data:
        df = pd.DataFrame(all_extracted_data)
        df = df[['company_name', 'stock_name', 'filing_time', 'new_product', 'product_description']]

        # Save using the exact pipe separator format requested by the professor
        df.to_csv("8k_product_releases.csv", index=False, sep='|')
        print(f"\nSuccessfully extracted {len(df)} product releases to 8k_product_releases.csv")
    else:
        print("\nNo new product releases found in the specified documents.")