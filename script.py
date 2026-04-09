from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from bs4 import BeautifulSoup
import re
import json

print("Starting script...")

# STEP 1: Load PDF
loader = PyPDFLoader("kb.pdf")
pages = loader.load()
text = " ".join([p.page_content for p in pages])

# STEP 2: Extract URLs
urls = re.findall(r'https?://\S+', text)
urls = list(set(urls))
print(f"Found {len(urls)} URLs")

# STEP 3: Fetch content from URLs
def get_text(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        # Remove unwanted elements
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        return soup.get_text(separator=" ", strip=True)
    except:
        return ""

documents = []

for url in urls:
    content = get_text(url)
    if content:
        documents.append({
            "url": url,
            "content": content
        })

print(f"Fetched {len(documents)} pages")

# STEP 4: Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

final_chunks = []

for doc in documents:
    chunks = splitter.split_text(doc["content"])
    for chunk in chunks:
        final_chunks.append({
            "text": chunk,
            "source": doc["url"]
        })

# STEP 5: Save output
with open("chunks_enterprise.json", "w", encoding="utf-8") as f:
    json.dump(final_chunks, f, indent=2)

print("Done! chunks_enterprise.json created.")