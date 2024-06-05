from typing import Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
import aiohttp
import pickle
import faiss
import os

ROOT = Path(__file__).resolve().parents[1]


class FAISSWrapper:
    """A wrapper class for handling FAISS indexing and querying with OpenAI embeddings."""

    def __init__(self, openai_key: str):
        self.openai_key = openai_key
        self.index_path = (ROOT / "query/faiss_index.index").resolve()
        self.index_ids_path = (ROOT / "query/faiss_index_ids.pkl").resolve()
        self.chunked_texts_path = (ROOT / "data/chunked_texts_ada.pkl").resolve()
        self.chunked_texts_ada = self.load_chunked_texts()
        self.index, self.ids = self.load_faiss_index()

    def load_chunked_texts(self) -> Dict[str, Dict[str, Any]]:
        """Load chunked texts with ADA embeddings from a pickle file."""
        with self.chunked_texts_path.open("rb") as f:
            return pickle.load(f)

    def extract_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """Extract embeddings and their corresponding IDs from the loaded chunked texts.

        Returns:
            A tuple containing a NumPy array of embeddings and a list of IDs.
        """
        embeddings = [value["ada_embedding"] for value in self.chunked_texts_ada.values()]
        ids = list(self.chunked_texts_ada.keys())
        return np.array(embeddings).astype("float32"), ids

    def create_faiss_index(self, embeddings: np.ndarray, ids: List[str]) -> faiss.IndexFlatIP:
        """Create and save a FAISS index from embeddings and IDs.

        Args:
            embeddings: A NumPy array of embeddings.
            ids: A list of IDs corresponding to the embeddings.

        Returns:
            The created FAISS index.
        """
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(self.index_path))

        with self.index_ids_path.open("wb") as f:
            pickle.dump(ids, f)

        return index

    def load_faiss_index(self) -> Tuple[faiss.IndexFlatIP, List[str]]:
        """Load the FAISS index and IDs, creating them if they do not exist.

        Returns:
            A tuple containing the FAISS index and the list of IDs.
        """
        if not self.index_path.exists() or not self.index_ids_path.exists():
            embeddings, ids = self.extract_embeddings()
            self.create_faiss_index(embeddings, ids)

        index = faiss.read_index(str(self.index_path))
        with self.index_ids_path.open("rb") as f:
            ids = pickle.load(f)
        return index, ids

    def query_faiss_index(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Query the FAISS index to find the top-k closest embeddings.

        Args:
            query_embedding: A NumPy array representing the query embedding.
            k: The number of top results to return.

        Returns:
            A list of dictionaries containing the top-k results.
        """
        D, I = self.index.search(query_embedding, k)
        top_k_ids = [self.ids[i] for i in I[0]]
        return [self.chunked_texts_ada[i] for i in top_k_ids]

    async def send_query(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Send a text query to OpenAI API to get embeddings and query the FAISS index.

        Args:
            text: The query text.
            k: The number of top results to return.

        Returns:
            A list of dictionaries containing the top-k results.
        """
        api_url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json",
        }
        data = {"input": [text], "model": "text-embedding-3-small"}

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=data) as response:
                if response.status == 200:
                    response_data = await response.json()
                    embeddings = np.array(response_data["data"][0]["embedding"]).astype("float32").reshape(1, -1)
                else:
                    response_data = await response.text()
                    raise Exception(f"Failed to get embeddings: {response_data}")

        return self.query_faiss_index(embeddings, k)


# Example usage
if __name__ == "__main__":
    indexer = FAISSWrapper(openai_key=os.environ.get("OPENAI_KEY"))

    sample_query = np.random.random((1, indexer.index.d)).astype("float32")  # Replace with actual query embedding
    top_k_results = indexer.query_faiss_index(sample_query, k=5)

    chunks = [k["chunk"] for k in top_k_results]
    for c in chunks:
        print(f"{c}\n\n\n\n")
