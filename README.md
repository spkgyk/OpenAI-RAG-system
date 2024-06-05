# Implementing Retrieval-Augmented Generation (RAG) on PDF Documents Using OpenAI and LLaMA 3

## How to Run

### Setting Up the Environment

1. **Set Environment Variable:**
   Set an environment variable for your OpenAI key named `OPENAI_KEY`.

   ```sh
   export OPENAI_KEY=your_openai_key_here
   ```

2. **Install Dependencies:**
   Run the setup script to install all dependencies in a conda environment and download LLaMA 3. Note that a GPU is required for FAISS and running LLaMA 3 8b Instruct locally.

   ```sh
   source src/setup/setup.sh
   ```

3. **Prepare the Data:**
   Navigate to the `src/data` directory and run the `pickle_data.py` script.

   ```sh
   cd src/data
   python pickle_data.py
   ```

4. **Run the Application:**
   Navigate to the `src` folder and run the RAG application with the following command. Replace `<model_name>` with the desired OpenAI model name or `"llama3"`, and `<k>` with the number of results to return from the RAG query. If the model argument is not passed, it defaults to the configuration found at `src/configs/default.yaml`.

   ```sh
   cd src
   python rag.py --model <model_name> --top-k <k>
   ```

## How It Works

The `src/data/pickle_data.py` script chunks the text and converts it into embeddings for dense retrieval via FAISS. When you ask the RAG application a question (by typing in the terminal), the query is converted into an embedding, and the inner product (dot product) is used to assess similarity via FAISS. The top 5 results are appended to the prompt and then given to the specified model. Based on my tests, `gpt-3.5-turbo` provides good results, while `gpt-4o` offers the best results.

## Future Improvements

1. **Text Quality:**
   The quality of text extracted from PDFs can be improved. Equations often become nearly unreadable and can be misleading.

2. **Retrieval System Quality:**
   Currently, chunks are split based on length with an overlap. Grouping text more logically, using semantic separation to group related content into single chunks, could improve retrieval quality. For instance, in the car PDF, separate instructions sometimes appear in a single chunk, reducing retrieval quality.

3. **Summarization of Chunks:**
   Using an LLM to summarize each chunk of data and using that summary as the embedding could enhance search results. 

4. **Metadata Inclusion:**
   Including metadata in the search process could improve the relevance and accuracy of the results.

5. **Dense-Sparse Retrieval System:**
   Implementing a hybrid retrieval system that combines dense retrieval with sparse retrieval (e.g., using BM25 alongside dense embeddings) could improve search performance. For instance, a weighted combination approach using the parameter \( \lambda \) to balance sparse and dense retrievals, such as \( \lambda \cdot \text{sparse} + (1 - \lambda) \cdot \text{dense} \), can be effective.

6. **Knowledge Graph Creation:**
   Using an LLM to create a knowledge graph could greatly enhance search results. For example, in the sentence "The queen died on September 8, 2022, at Balmoral Castle in Scotland," we could create connections between September 8, the queen, and Balmoral Castle in Scotland. This approach could lead to more accurate search results but may increase latency.

7. **Testing and Quality Assurance:**
   Currently, there are no unit tests in place to verify the functionality of our system. In addition, to enhance our testing framework, we could implement `gpt-4o` to read and evaluate the quality of our Retrieval-Augmented Generation (RAG) system after each user-assistant message pair.

8. **Optimization and Alignment:**
   We could utilize `gpt-4o` to provide improved responses, which can be saved for policy optimization and alignment through [ORPO](https://arxiv.org/abs/2403.07691) (Odds Ratio Preference Optimization) or [DPO](https://arxiv.org/abs/2305.18290) (Direct Preference Optimization).