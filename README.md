# 🧠 Query-Expanded Retrieval-Augmented Generation (RAG) on PDF Documents

This project implements a **Query Expansion-based Retrieval-Augmented Generation (RAG)** pipeline using LangChain, Hugging Face Transformers, and ChromaDB. It demonstrates how to enhance context retrieval from a PDF (e.g., a Microsoft Annual Report) by expanding a single user query into multiple semantically relevant sub-queries.

---

## 🔧 What This Project Does

✅ Parses and loads text from a PDF  
✅ Splits content into sentence-aware token chunks  
✅ Embeds and stores chunks using `sentence-transformers/all-MiniLM-L6-v2` in ChromaDB  
✅ Generates query expansions using an LLM (`DeepSeek-R1-Distill-Qwen-7B`)  
✅ Retrieves top-k relevant chunks for each expanded query  
✅ Displays the matched context chunks for each query  

---

## 📁 File Structure

