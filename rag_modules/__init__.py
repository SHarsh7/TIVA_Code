"""Modular RAG package for the EV-bus intelligence pipeline.

Modules: data_ingestion (corpus + chunking), embedding_engine (MiniLM/FAISS),
retrieval_tuning (hyperparameter grid, Hit@5/MRR@10), rag_orchestration
(hybrid retrieval + rerank + grounded generation).
"""
