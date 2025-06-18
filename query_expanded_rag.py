import os
from pypdf import PdfReader
from langchain_chroma import Chroma
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
reader = PdfReader("data/microsoft-annual-report.pdf")

pdf_texts = [p.extract_text().strip() for p in reader.pages]

pdf_texts = [text for text in pdf_texts if text]

character_splitter = RecursiveCharacterTextSplitter(
    separators = ["\n\n", "\n", ". ", " ", ""],
    chunk_size = 1000,
    chunk_overlap = 0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap = 0,
    tokens_per_chunk=256
)

token_split_texts = []
for text in character_split_texts:
  token_split_texts += token_splitter.split_text(text)

microsoft_db = Chroma(
    collection_name = "microsoft_db",
    embedding_function = embedding_model
)

ids = [str(i) for i in range(len(token_split_texts))]

microsoft_db.add_documents(documents= token_split_texts, ids=ids)

retriever = microsoft_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# query = "What was the total revenue for the year?"
# results_docs = retriever.get_relevant_documents(query)

# results = []
# for doc in result_docs:
#   results.append(doc.page_content)

def generate_multi_query(query, model):
    prompt = """
        You are a knowledgeable financial research assistant. 
        Your users are inquiring about an annual report. 
        For the given question, propose up to five related questions to assist them in finding the information they need. 
        Provide concise, single-topic questions (withouth compounding sentences) that cover various aspects of the topic. 
        Ensure each question is complete and directly related to the original inquiry. 
        List each question on a separate line without numbering.
        """
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": query},
    ]
    pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    response = pipe(messages)
    content = response[0]["generated_text"].split("\n")
    return content

original_query = "What details can you provide about the factors that led to revenue growth?"
aug_queries = generate_multi_query(original_query)

joint_query = [original_query] + aug_queries

result_docs = []
for i in range(len(joint_query)):
  results_docs.append(retriever.get_relevant_documents(joint_query[i]))

results = []
for i in range(len(joint_query)):
  doc_contents = [doc.page_content for doc in result_docs[i]]
  results[i].append(doc_contents)

for i, documents in enumerate(results):
    print(f"Query: {joint_query[i]}")
    print("")
    print("Results:")
    for doc in documents:
        print(word_wrap(doc))
        print("")
    print("-" * 100)