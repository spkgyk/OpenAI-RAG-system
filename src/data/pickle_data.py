from typing import Union, List, Dict, Any, Callable
from pdfminer.high_level import extract_text
from nltk.tokenize import TweetTokenizer
from tqdm.asyncio import tqdm
from pathlib import Path
import asyncio
import aiohttp
import pickle
import re
import os

openai_key = os.environ.get("OPENAI_KEY")


class CustomTweetTokenizer(TweetTokenizer):
    def tokenize_with_spaces(self, text):
        # Use the parent class's tokenize method to tokenize the text
        tokens = super().tokenize(text)

        # Reconstruct the text with preserved characters
        result = []
        i = 0
        for token in tokens:
            while i < len(text) and text[i] != token[0]:
                result.append(text[i])
                i += 1
            result.append(token)
            i += len(token)

        # Add remaining characters at the end
        while i < len(text):
            result.append(text[i])
            i += 1

        return result


def clean_text(text: str) -> str:
    """Clean text."""
    text = re.sub(r"\n(?!\n)\s*", " ", text)
    text = re.sub(r"\n\s*", "\n\n", text)
    return text.strip()


def to_markdown(pdf_path: Union[Path, str], output_path: Union[Path, str]) -> Dict[str, List[str]]:
    """Convert PDF files to Markdown and clean the text.

    Args:
        pdf_path: Path to the PDF files.
        output_path: Path to save the converted Markdown files.

    Returns:
        A dictionary with filenames as keys and tokenized text as values.
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)

    if not output_path.exists():
        output_path.mkdir(parents=True)

    md_texts = {}
    tokenizer = CustomTweetTokenizer()

    for root, _, files in os.walk(pdf_path):
        for file in files:
            output_file_name = output_path / file.replace(".pdf", ".txt")
            if not output_file_name.exists():
                md_text = extract_text(os.path.join(root, file))
                md_text = clean_text(md_text)
                output_file_name.write_bytes(md_text.encode())
            else:
                with output_file_name.open() as f:
                    md_text = f.read()

            md_texts[file] = tokenizer.tokenize_with_spaces(md_text)

    return md_texts


def split_text(md_texts: Dict[str, List[str]], chunk_size: int = 400, chunk_overlap: int = 40) -> Dict[str, Dict[str, Any]]:
    """Split text into chunks with specified size and overlap.

    Args:
        md_texts: Dictionary with filenames as keys and tokenized text as values.
        chunk_size: Number of tokens per chunk.
        chunk_overlap: Number of overlapping tokens between chunks.

    Returns:
        A dictionary with chunk information including the chunk text and IDs.
    """
    chunked_texts = {}
    for file_name, md_text in md_texts.items():
        chunks = {}
        start = 0
        counter = 0
        text_length = len(md_text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = md_text[start:end]
            chunk_id = f"{file_name}_chunk_{counter}"
            chunks[chunk_id] = {
                "chunk": "".join(chunk),
                "chunk_id": chunk_id,
                "chunk_number": counter,
            }
            start += chunk_size - chunk_overlap
            counter += 1

        chunked_texts.update(chunks)

    return chunked_texts


async def fetch_embeddings(session: aiohttp.ClientSession, batch: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    """Fetch embeddings for a batch of text chunks using OpenAI API.

    Args:
        session: An active aiohttp session.
        batch: List of text chunks with chunk IDs.
        model: Model name to use for embedding.

    Returns:
        A list of dictionaries with chunk IDs and their corresponding embeddings.
    """
    api_url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json",
    }

    chunks = [item["chunk"] for item in batch]
    keys = [item["chunk_id"] for item in batch]

    data = {"input": chunks, "model": model}
    async with session.post(api_url, headers=headers, json=data) as response:
        response_data = await response.json()
        return [{"chunk_id": key, "embedding": item["embedding"]} for key, item in zip(keys, response_data["data"])]


async def get_all_embeddings(
    chunked_texts: Dict[str, Dict[str, Any]],
    model: str = "text-embedding-3-small",
    batch_size: int = 20,
) -> Dict[str, Dict[str, Any]]:
    """Get embeddings for all text chunks asynchronously.

    Args:
        chunked_texts: Dictionary with chunk information.
        model: Model name to use for embedding.
        batch_size: Number of text chunks to process per batch.

    Returns:
        A dictionary with updated chunk information including embeddings.
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        text_items = list(chunked_texts.values())
        for i in range(0, len(text_items), batch_size):
            batch = text_items[i : i + batch_size]
            tasks.append(fetch_embeddings(session, batch, model))

        embeddings = {}
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            embeddings_batch = await task
            for item in embeddings_batch:
                embeddings[item["chunk_id"]] = item["embedding"]

        for key, embedding in embeddings.items():
            chunked_texts[key]["ada_embedding"] = embedding

        return chunked_texts


def get_ada(
    chunked_texts: Dict[str, Dict[str, Any]],
    func: Callable[[Dict[str, Dict[str, Any]], str, int], Any],
    model: str = "text-embedding-3-small",
    batch_size: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """Run the embedding function and return the updated chunked texts.

    Args:
        chunked_texts: Dictionary with chunk information.
        func: Function to get embeddings.
        model: Model name to use for embedding.
        batch_size: Number of text chunks to process per batch.

    Returns:
        A dictionary with updated chunk information including embeddings.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    embeddings = asyncio.run(func(chunked_texts, model, batch_size))
    return embeddings


if __name__ == "__main__":
    md_texts = to_markdown("./pdfs", "./md_pdfs")
    chunked_texts = split_text(md_texts, 400, 80)
    chunked_texts_ada = get_ada(chunked_texts, get_all_embeddings, "text-embedding-3-small", 5)

    with open("chunked_texts_ada.pkl", "wb") as f:
        pickle.dump(chunked_texts_ada, f)
