**Project: Jupiter FAQ Bot**

---

## Overview

The Jupiter FAQ Bot is a semantic-search driven chatbot that answers user questions about Jupiter’s mobile banking app. It integrates Azure OpenAI to generate embeddings and chat completions via REST API calls, uses FAISS for efficient similarity search, and presents a friendly, multilingual experience through a Streamlit UI.

**Key Capabilities:**

* Semantic retrieval of the most relevant FAQs.
* Natural language answer generation with Azure OpenAI chat completions.
* Graceful decline when confidence is low.
* Multilingual support (English, Hindi, Hinglish).
* Related question suggestions to guide users.

---

## Architecture & Methodology

### 1. Data Layer

* **Source:** `data/faqs.json` contains categorized Q\&A pairs under topics like KYC, Rewards, Payments, Limits, Security.
* **Flattening:** At index time, the nested JSON is flattened into a list of `{ question, answer, category }` for uniform processing.

### 2. Embedding Pipeline

1. **Model Selection:** Uses an Azure-deployed embedding model (e.g., `text-embedding-ada-002`).
2. **REST API Call:** The script `index_faqs.py` makes POST requests to the `/embeddings` endpoint.
3. **Vector Storage:** Responses yield 1536-dimensional vectors, aggregated into a NumPy array.
4. **Indexing:** FAISS `IndexFlatL2` stores the vectors for fast nearest-neighbor search.
5. **Metadata:** Alongside the FAISS index, `embeddings/metadata.json` saves the flattened FAQ entries for lookup.

### 3. Retrieval & Matching

* **Query Embedding:** On a user query, `faq_bot.py` sends the text to the same embedding endpoint.
* **Similarity Search:** FAISS returns the top-K nearest stored FAQ vectors and their distances.
* **Thresholding:** If the top distance exceeds a set threshold (e.g., 0.5), the bot declines to answer confidently.

### 4. Chat Completion

* **Prompt Construction:** The bot creates a context-aware prompt including:

  * System message (role definition).
  * User query.
  * Retrieved best-match Q\&A as context.
  * Language specification.
* **REST API Call:** Calls the `/chat/completions` endpoint via `requests`.
* **Response Parsing:** Extracts the generated answer text.

### 5. Frontend (Streamlit)

* **User Interface:** Simple input box for queries, language selector, and submit button.
* **Display:** Shows bot response and source FAQ question.
* **Related Suggestions:** Optionally lists semantically close questions for further exploration.

---

## File Structure

```text
jupiter-faq-bot/
├── data/
│   └── faqs.json               # Raw categorized FAQs
├── embeddings/
│   ├── faiss_index.idx         # FAISS index file
│   └── metadata.json           # Flattened FAQ entries
├── index_faqs.py               # Builds embeddings and FAISS index
├── faq_bot.py                  # Embedding, retrieval, chat logic
├── app.py                      # Streamlit frontend simulation
├── README.md                   # This documentation
└── requirements.txt            # Python dependencies
```

---

## Setup & Installation

1. **Clone Repository**

   ```bash
   git clone https://github.com/your-org/jupiter-faq-bot.git
   cd jupiter-faq-bot
   ```

2. **Python Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   Create a `.env` file in the project root with:

   ```ini
   AZURE_OPENAI_API_KEY=<your-key>
   AZURE_OPENAI_ENDPOINT=<https://your-resource.openai.azure.com/>
   AZURE_OPENAI_VERSION=2023-05-15
   AZURE_DEPLOYMENT=<your-chat-deployment>
   AZURE_OPENAI_DEPLOYMENT_EMBEDDING=<your-embedding-deployment>
   ```

4. **Build Embeddings**

   ```bash
   python index_faqs.py
   ```

5. **Run the App**

   ```bash
   streamlit run app.py
   ```

---

## Future Enhancements

* **Offline Embeddings:** Integrate local Hugging Face models for embedding generation.
* **Advanced Reranking:** Use cross-encoder ranking for more precise answers.
* **Contextual Memory:** Store past user interactions for multi-turn dialog continuity.
* **Analytics Dashboard:** Track user queries and bot performance metrics.

---

## License & Contributions

Licensed under MIT. Contributions welcome via pull requests or issues on GitHub.
