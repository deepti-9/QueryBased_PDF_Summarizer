import fitz
import re
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Disable MuPDF noisy logs
fitz.TOOLS.mupdf_display_errors(False)


class PDFProcessor:
    def __init__(self, llm=None, vision_func=None, process_images=False):
        self.llm = llm
        self.vision_func = vision_func
        self.process_images = process_images

        # Optimized chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    # Improved author extraction
    def extract_authors(self, text: str) -> str:
        import re

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        lines = lines[:20]

        for line in lines:

            if "abstract" in line.lower():
                break

            if "," in line or " and " in line.lower():

                # Remove superscripts
                line = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]", "", line)
                line = re.sub(r"\(?\d+\)?", "", line)

                # Remove designations
                line = re.sub(r"\(.*?\)", "", line)

                # Normalize AND
                line = line.replace("AND", ",").replace("and", ",")

                # Clean spacing
                line = re.sub(r"\s+", " ", line).strip()

                # Handle ALL CAPS
                if line.isupper():
                    line = line.title()

                # Extract names (supports initials)
                names = re.findall(
                    r"\b(?:[A-Z]\.\s*)?[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b",
                    line
                )

                # Final validation
                names = [
                    n for n in names
                    if len(n.split()) >= 2 and not any(char.isdigit() for char in n)
                ]

                if names:
                    return ", ".join(names)

        return "Unknown Author"
    # Fixed year extraction
    def extract_year(self, text: str) -> str:
        lines = text.lower().split("\n")

        # Priority lines
        for line in lines[:40]:
            if "received" in line or "published" in line:
                match = re.search(r'(20\d{2})', line)
                if match:
                    return match.group(1)

        # fallback
        years = re.findall(r'(20\d{2})', text)
        if years:
            return years[0]

        return "N/A"

    # Clean text instead of skipping
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove DOI
        text = re.sub(r'doi:\S+', '', text)

        # Remove reference section ONLY
        text = re.sub(r'References.*', '', text, flags=re.IGNORECASE)

        return text.strip()

    def process_pdfs(self, pdf_directory: str) -> List[Document]:
        all_docs = []
        pdf_files = list(Path(pdf_directory).glob("**/*.pdf"))

        for pdf_file in pdf_files:
            try:
                with open(pdf_file, "rb") as f:
                    pdf_bytes = f.read()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            except Exception as e:
                print(f" Skipping file {pdf_file}: {e}")
                continue

            # Extract metadata
            try:
                first_page_text = doc[0].get_text()
            except:
                first_page_text = ""

            author = self.extract_authors(first_page_text[:2000])
            year = self.extract_year(first_page_text)

            # Process pages
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                except:
                    continue

                base_metadata = {
                    "source_file": pdf_file.name,
                    "page": page_num + 1,
                    "author": author,
                    "date": year,
                    "type": "text"
                }

                try:
                    text_blocks = page.get_text("blocks", sort=True)
                    full_text = "\n\n".join(
                        b[4].strip() for b in text_blocks if b[4].strip()
                    )
                except:
                    continue

                if not full_text or len(full_text.strip()) < 80:
                    continue

                # CLEAN instead of SKIP
                full_text = self.clean_text(full_text)

                all_docs.append(Document(
                    page_content=full_text,
                    metadata=base_metadata
                ))

            doc.close()

        return all_docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        final_chunks = []

        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            final_chunks.extend(chunks)

        return final_chunks