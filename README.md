# Gottfried
The application is a Q&A chatbot (Gottfried) that uses a database of documents to retrieve information. The documents are loaded and split into chunks, which are then embedded using OpenAI embeddings. The embedding vectors are stored in Pinecone DB, a vector database that is designed to store, index, and search high-dimensional vector embeddings at scale.

When a user enters a prompt, the application queries the top 5 vectors based on the similarity with the prompt (dotproduct). The prompt is then sent to the OpenAI API, and a LLM (GPT-3.5-turbo) provides an answer based on the top 5 vectors retrieved from Pinecone. The chatbot provides an efficient and accurate way to retrieve information from a large database of documents.
