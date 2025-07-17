import requests
import tempfile
import os
import re
import logging
from arxiv2text import arxiv_to_text, arxiv_to_md
from .utils import fix_markdown_headers, chunk_markdown

logger = logging.getLogger(__name__)

def fetch_and_chunk_arxiv_full_text(arxiv_link):
    """
    Fetches the full text of a paper from Arxiv given an access URL and returns it as a list of chunks.
    
    Args:
        arxiv_link (str): The link to the arxiv paper.
        
    Returns:
        list: A list of text chunks if available, otherwise an error message.
    """
    
    try:
        # Replace /abs/ with /pdf/ to get the PDF link
        pdf_url = arxiv_link.replace("/abs/", "/pdf/")
        print(pdf_url)

        # Retrieve markdown text
        arxiv_paper_text = fetch_full_text_to_memory(arxiv_link)
        if not arxiv_paper_text:
            logger.info(f"❌ Failed to fetch markdown text for {arxiv_link}")
            return []
        # Trim it down to actual content (Introduction - before References/Bibliography)
        arxiv_paper_text_clean = extract_main_content(arxiv_paper_text)
        # Mark section headers with markdown syntax
        fixed_md = fix_markdown_headers(arxiv_paper_text_clean)
        # Convert the cleaned markdown text to chunks
        chunked_md = chunk_markdown(fixed_md, max_tokens = 400)
        
        return chunked_md

    except requests.RequestException as e:
        return f"Error fetching paper: {e}"


def fetch_full_text_from_arxiv(arxiv_link, output_folder):
    """
    Fetches the full text of a paper from Arxiv given an access URL.
    
    Args:
        arxiv_link (str): The link to the arxiv paper.
        
    Returns:
        str: The full text of the paper if available, otherwise an error message.
    """
    
    try:

        # Replace /abs/ with /pdf/ to get the PDF link
        pdf_url = arxiv_link.replace("/abs/", "/pdf/")
        logger.info(f"Fetching full text from {pdf_url}")

        # Try to retrieve markdown directly if that doesnt work, fallback to normal text string
        full_text = ""
        full_text = arxiv_to_md(pdf_url, output_folder)
        
        if not full_text:
            logger.info(f"❌ Failed to fetch markdown text for {arxiv_link}")
            logger.info(f"Try to fetch normal text instead.")

            full_text = arxiv_to_text(pdf_url)
                
        return full_text
    except requests.RequestException as e:
        return f"Error fetching paper: {e}"
    except Exception as e:
        logger.warning(f"Failed to extract markdown from {pdf_url}: {e}")
        return ""  # or return None, or skip this paper
    

def fetch_full_text_to_memory(paper_id):
    """
    Fetches the full text as markdown, writes it to a temp file, reads it back into memory, and returns the string.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        md_text = fetch_full_text_from_arxiv(paper_id, output_folder=tmpdirname)
        if not isinstance(md_text, str):
            # arxiv_to_md probably writes to a file, so let's read it
            # Find the first .md file in tmpdirname
            md_files = [f for f in os.listdir(tmpdirname) if f.endswith('.md')]
            if not md_files:
                return "Failed to fetch markdown text."
            tmp_path = os.path.join(tmpdirname, md_files[0])
            with open(tmp_path, "r", encoding="utf-8") as f:
                loaded_md = f.read()
            return loaded_md
        else:
            return md_text
        

def extract_main_content(text):
    """
    Extracts the main content of a paper, starting from the Introduction section
    and ending before References/Bibliography.
    """
    # Case-insensitive search for 'introduction'
    intro_match = re.search(r'(?i)\bintroduction\b', text)
    if not intro_match:
        return "Introduction section not found."
    start_idx = intro_match.start()

    # Case-insensitive search for 'references' or 'bibliography' after introduction
    refs_match = re.search(r'(?i)\b(references|bibliography)\b', text[start_idx:])
    if refs_match:
        end_idx = start_idx + refs_match.start()
    else:
        end_idx = len(text)

    return text[start_idx:end_idx].strip()

# if __name__ == "__main__":
#     LOCAL TESTING ONLY: The following block is for manual/local testing and should not be run in production or on import.
#     arxiv_link = "https://arxiv.org/abs/1706.03762"
#     chunks = fetch_and_chunk_arxiv_full_text(arxiv_link)
#     for i, chunk in enumerate(chunks):
#         print(f"Chunk {i+1}:\n{chunk}\n")
