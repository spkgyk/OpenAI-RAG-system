from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms.ollama import Ollama
from configs import ConfigLoader
from openai import AsyncOpenAI
from query import FAISSWrapper
from typing import Dict, Any
import asyncio

PROMPT_TEMPLATE = """
Answer the question, using the context below as a guide.

Context:
---
{}
---

Answer the question based on the above context: "{}"
If you use any of the information in the context, cite the source (i.e. Sources: ..., ..., ...)
"""


class RAG:
    """A class for performing Retrieval-Augmented Generation (RAG) with different language models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.wrapper = FAISSWrapper(config["openai_key"])

        if config["model"] == "llama3":
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            self.llama3_client = Ollama(model="llama3", callback_manager=callback_manager)
            self.llama_messages = []
        elif "gpt" in config["model"]:
            self.model_name = config["model"]
            self.gpt_client = AsyncOpenAI(api_key=config["openai_key"])
            self.gpt_messages = []

    async def _get_prompt(self, query: str, k: int = 5) -> str:
        """Generate the prompt for the language model using context retrieved from FAISS.

        Args:
            query: The input query string.
            k: The number of top results to retrieve.

        Returns:
            A formatted prompt string.
        """

        def format_chunk(i: int, chunk: Dict[str, Any]) -> str:
            return f"""Source {i}: "{chunk["chunk_id"]}"\nInformation {i}: "{chunk["chunk"]}\""""

        chunks = await self.wrapper.send_query(query, k)
        context = "\n\n".join(format_chunk(i, chunk) for i, chunk in enumerate(chunks))
        return PROMPT_TEMPLATE.format(context, query)

    async def _ask_llama3(self, query: str, k: int = 5) -> None:
        """Ask a query using the Llama3 model and handle the response.

        Args:
            query: The input query string.
            k: The number of top results to retrieve.
        """
        prompt = await self._get_prompt(query, k)
        self.llama_messages.append(HumanMessage(content=prompt))
        response = await self.llama3_client.ainvoke(self.llama_messages)
        self.llama_messages.pop()
        self.llama_messages.append(HumanMessage(content=query))
        self.llama_messages.append(AIMessage(content=response))

    async def _ask_gpt(self, query: str, k: int = 5) -> None:
        """Ask a query using the GPT model and handle the response.

        Args:
            query: The input query string.
            k: The number of top results to retrieve.
        """
        prompt = await self._get_prompt(query, k)
        self.gpt_messages.append({"role": "user", "content": prompt})
        stream = await self.gpt_client.chat.completions.create(
            messages=self.gpt_messages,
            model=self.model_name,
            stream=True,
        )
        streamed_chunks = []
        async for item in stream:
            chunk = item.choices[0].delta.content or ""
            streamed_chunks.append(chunk)
            print(chunk, end="", flush=True)
        self.gpt_messages.pop()
        self.gpt_messages.append({"role": "user", "content": query})
        self.gpt_messages.append({"role": "assistant", "content": "".join(streamed_chunks)})

    async def ask(self, query: str, k: int = 5) -> None:
        """Ask a query using the configured model and print the response.

        Args:
            query: The input query string.
            k: The number of top results to retrieve.
        """
        print(f"{self.config['model']}: ", end="", flush=True)
        try:
            if self.config["model"] == "llama3":
                await self._ask_llama3(query, self.config["top_k"])
            elif "gpt" in self.config["model"]:
                await self._ask_gpt(query, self.config["top_k"])
        except Exception as e:
            print(f"Error chatting with {self.config['model']}: {e}")
        finally:
            print("\n----------------------------------\n")


async def main():
    """Main function to run the RAG terminal application."""
    print("Welcome to the RAG Terminal Application!")
    print("Type 'exit' to quit.")

    config_loader = ConfigLoader("configs/default.yaml")
    config = config_loader.get_config()
    rag = RAG(config)
    print(f"Using model: {config['model']}")

    while True:
        question = input("You: ")

        if question.lower() == "exit":
            print("Goodbye!", end="")
            break

        await rag.ask(question)


if __name__ == "__main__":
    asyncio.run(main())
