# LlamaIndex Crash Course: Beginner to Advanced RAG App

A practical end-to-end LlamaIndex project that demonstrates how to build a Retrieval-Augmented Generation (RAG) application using Python, OpenRouter, HuggingFace embeddings, BM25, reranking, query routing, sub-question decomposition, and Gradio.

This project is designed as a hands-on notebook for learning and demonstrating real AI engineering workflows: document loading, chunking, embedding, indexing, retrieval, answer generation, source inspection, evaluation, and deployment through a simple web app.

## Links

- YouTube tutorial: https://youtu.be/NLIEVgoeF-s
- Kaggle notebook: https://www.kaggle.com/code/yisberh/llamaindex-crash-course-beginner-advanced

## Project Overview

The main goal of this project is to show how a document question-answering system works from the inside.

Instead of only calling an LLM directly, the system first searches relevant documents, retrieves the most useful context, and then asks the LLM to generate an answer based on that context.

High-level flow:

```text
Documents
↓
Chunks / Nodes
↓
Embeddings
↓
Vector Index
↓
Retriever
↓
LLM
↓
Answer + Sources
```

The final result is a small AI Study Assistant that can upload TXT/PDF documents, build an index, ask questions, return answers, and show the source chunks used by the model.

## What This Project Demonstrates

This project covers the full RAG pipeline:

- Loading TXT and PDF documents
- Splitting documents into chunks
- Creating embeddings with HuggingFace models
- Building a vector index with LlamaIndex
- Asking questions using a query engine
- Inspecting retrieved chunks and source nodes
- Controlling chunk size and chunk overlap
- Using custom prompts to reduce unsupported answers
- Saving and reloading indexes
- Building a chat engine for follow-up questions
- Running simple answer evaluation
- Creating tools and agents
- Using hybrid search with BM25 + vector retrieval
- Comparing vector, keyword, and hybrid retrieval
- Applying reranking to improve retrieved context
- Rewriting vague user questions
- Routing questions to the right query engine
- Using sub-question decomposition for complex questions
- Building a Gradio web app

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| LlamaIndex | RAG framework |
| OpenRouter | LLM API access |
| HuggingFace Embeddings | Local/free embedding model |
| BM25 | Keyword-based retrieval |
| SentenceTransformers | Reranking model |
| Gradio | Web app interface |
| Kaggle Notebook | Development environment |
| TXT/PDF files | Input documents |

## Dataset Used

The notebook uses a simple fake school report dataset to make the RAG pipeline easy to understand.

The dataset includes:

```text
grade_5_report.txt
grade_6_report.txt
grade_7_report.txt
grade_8_report.txt
```

Each grade report includes:

- Student name
- Gender
- English score
- Mathematics score
- Science score
- Social Studies score
- Attendance
- Final average
- Student status
- Grade-level summary

Example questions:

```text
Who is the top student in Grade 8?
Which students need support in Mathematics?
Which grade has the highest class average?
How many female students are in Grade 8?
What is the school lunch menu?
```

The last question is intentionally not included in the documents, so the notebook can test whether the system avoids unsupported answers.

---

## 1. Installation

The notebook installs the required packages:

```python
!pip install -q llama-index llama-index-llms-openai \
llama-index-embeddings-huggingface sentence-transformers pypdf gradio
```

For advanced retrieval:

```python
!pip install -q llama-index-retrievers-bm25 rank-bm25 sentence-transformers
```

---

## 2. OpenRouter API Setup

The project uses OpenRouter as the LLM provider.

In Kaggle, the API key is loaded from Kaggle Secrets.

Secret name:

```text
OPENROUTER_API_KEY
```

Example setup:

```python
import os

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    os.environ["OPENROUTER_API_KEY"] = user_secrets.get_secret("OPENROUTER_API_KEY")
    print("OpenRouter API key loaded.")
except Exception:
    print("Could not load Kaggle Secret.")
```

---

## 3. LlamaIndex Configuration

This project uses:

- OpenRouter for LLM answer generation
- HuggingFace embeddings for vector search

```python
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

Settings.llm = OpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    api_base="https://openrouter.ai/api/v1",
    temperature=0.1,
)
```

---

## 4. Loading Documents

Documents are loaded using `SimpleDirectoryReader`.

```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()

print(f"Number of loaded documents: {len(documents)}")

for doc in documents:
    print(doc.metadata.get("file_name"))
```

Expected output:

```text
Number of loaded documents: 4
grade_5_report.txt
grade_6_report.txt
grade_7_report.txt
grade_8_report.txt
```

---

## 5. Building the Vector Index

The vector index converts documents into searchable embeddings.

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)

print("Vector index created.")
```

Internal flow:

```text
documents
↓
chunks
↓
embeddings
↓
vector index
```

---

## 6. Query Engine

The query engine retrieves relevant chunks and asks the LLM to generate an answer.

```python
query_engine = index.as_query_engine(similarity_top_k=3)

response = query_engine.query("Who is the top student in Grade 8?")

print(response)
```

Expected output:

```text
Betelhem Dawit is the top student in Grade 8 with a final average of 93.25.
```

---

## 7. Retrieval Inspection

The notebook includes retrieval-only inspection to verify what the system finds before the LLM generates the final answer.

```python
retriever = index.as_retriever(similarity_top_k=3)

results = retriever.retrieve("Which students need support in Grade 7?")

for i, node in enumerate(results, start=1):
    print(f"Result {i}")
    print("Score:", node.score)
    print("File:", node.metadata.get("file_name"))
    print(node.text[:500])
    print("-" * 80)
```

Expected retrieved file:

```text
grade_7_report.txt
```

Expected retrieved content includes:

```text
Yosef Fikru
Tinsae Solomon
Needs Support
Mathematics
```

---

## 8. Source Nodes

Source nodes show which chunks were used to answer a question.

```python
response = query_engine.query("Why does Grade 8 have strong performance?")

print("ANSWER:")
print(response)

print("\nSOURCE CHUNKS:")
for i, source in enumerate(response.source_nodes, start=1):
    print(f"Source {i}")
    print("Score:", source.score)
    print("File:", source.node.metadata.get("file_name"))
    print(source.node.text[:500])
    print("-" * 80)
```

This makes the RAG system easier to inspect and debug.

---

## 9. Chunking and Chunk Overlap

Chunking splits long documents into smaller nodes.

Chunk overlap helps preserve context when information is split between two chunks.

```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=256,
    chunk_overlap=30
)

nodes = splitter.get_nodes_from_documents(documents)

print(f"Number of chunks/nodes created: {len(nodes)}")
```

To inspect overlap:

```python
for i in range(len(nodes) - 1):
    print(f"\n========== Node {i+1} → Node {i+2} ==========")

    print("\nEND OF CURRENT NODE:")
    print(nodes[i].text[-250:])

    print("\nSTART OF NEXT NODE:")
    print(nodes[i + 1].text[:250])
```

---

## 10. Custom Index From Chunks

After creating custom nodes, the notebook builds an index from those nodes.

```python
custom_index = VectorStoreIndex(nodes)

custom_query_engine = custom_index.as_query_engine(
    similarity_top_k=3
)

response = custom_query_engine.query(
    "Which students need support in Mathematics?"
)

print(response)
```

Expected answer should mention students such as:

```text
Yonas Mengistu
Dawit Kebede
Yosef Fikru
Tinsae Solomon
Ermias Kebede
```

---

## 11. Custom Prompt

A custom prompt helps control model behavior and reduce unsupported answers.

```python
from llama_index.core import PromptTemplate

qa_prompt = PromptTemplate(
    """
You are a clear and helpful AI tutor.

Use only the context below to answer the question.
If the answer is not found in the context, say:
"I don't know based on the provided documents."

Context:
{context_str}

Question:
{query_str}

Answer:
"""
)

safe_query_engine = index.as_query_engine(
    similarity_top_k=3,
    text_qa_template=qa_prompt
)

response = safe_query_engine.query("What is the school lunch menu?")

print(response)
```

Expected output:

```text
I don't know based on the provided documents.
```

---

## 12. PDF Question Answering

The same RAG pipeline can be applied to PDF files.

```python
from pathlib import Path

pdf_dir = Path("pdf_data")
pdf_dir.mkdir(exist_ok=True)

print("Upload PDF files into:", pdf_dir.resolve())
```

After uploading PDFs:

```python
pdf_files = list(pdf_dir.glob("*.pdf"))

if not pdf_files:
    print("No PDF files found.")
else:
    pdf_documents = SimpleDirectoryReader("pdf_data").load_data()

    pdf_index = VectorStoreIndex.from_documents(pdf_documents)

    pdf_query_engine = pdf_index.as_query_engine(
        similarity_top_k=3,
        text_qa_template=qa_prompt
    )

    pdf_response = pdf_query_engine.query(
        "Summarize this PDF clearly."
    )

    print(pdf_response)
```

---

## 13. Save and Reload Index

Saving the index avoids rebuilding embeddings every time.

Save index:

```python
index.storage_context.persist(persist_dir="storage")

print("Index saved.")
```

Reload index:

```python
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(
    persist_dir="storage"
)

loaded_index = load_index_from_storage(storage_context)

loaded_query_engine = loaded_index.as_query_engine(
    similarity_top_k=3,
    text_qa_template=qa_prompt
)

response = loaded_query_engine.query(
    "Who is the top student in Grade 8?"
)

print(response)
```

Expected output:

```text
Betelhem Dawit is the top student in Grade 8 with a final average of 93.25.
```

---

## 14. Chat Engine

The chat engine supports follow-up questions.

```python
chat_engine = index.as_chat_engine(
    chat_mode="condense_question",
    verbose=True
)

reply1 = chat_engine.chat("Who is the top student in Grade 8?")
print("Reply 1:")
print(reply1)

reply2 = chat_engine.chat("What is the final average?")
print("\nReply 2:")
print(reply2)
```

Expected output:

```text
Reply 1:
Betelhem Dawit is the top student in Grade 8.

Reply 2:
Her final average is 93.25.
```

---

## 15. Simple Evaluation

The notebook includes a simple evaluation loop using expected keywords.

```python
test_cases = [
    {
        "question": "Who is the top student in Grade 7?",
        "expected_keyword": "Sara Abebe"
    },
    {
        "question": "Which grade has the highest class average?",
        "expected_keyword": "Grade 8"
    },
    {
        "question": "Which students need support in Grade 5?",
        "expected_keyword": "Yonas"
    }
]

for test in test_cases:
    response = safe_query_engine.query(test["question"])
    answer = str(response)

    passed = test["expected_keyword"].lower() in answer.lower()

    print("Question:", test["question"])
    print("Answer:", answer)
    print("Expected keyword:", test["expected_keyword"])
    print("Passed:", passed)
    print("-" * 80)
```

Expected output:

```text
Question: Who is the top student in Grade 7?
Expected keyword: Sara Abebe
Passed: True

Question: Which grade has the highest class average?
Expected keyword: Grade 8
Passed: True
```

---

## 16. Tools and Agents

The notebook includes a simple FunctionAgent example.

A tool is a Python function that the agent can call when needed.

```python
from llama_index.core.agent.workflow import FunctionAgent

def multiply(a: float, b: float) -> float:
    return a * b

agent = FunctionAgent(
    tools=[multiply],
    llm=Settings.llm,
    system_prompt="You are a helpful assistant. Use tools when they are useful.",
    streaming=False,
)

agent_response = await agent.run(
    "What is 1234 multiplied by 5678?"
)

print(agent_response)
```

Expected output:

```text
7006652
```

---

## 17. Hybrid Search

Hybrid search combines semantic vector search with BM25 keyword search.

```python
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

vector_retriever = custom_index.as_retriever(
    similarity_top_k=3
)

bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=3
)

hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=3,
    num_queries=1,
    mode="reciprocal_rerank",
    use_async=False,
    verbose=True,
)

hybrid_query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    text_qa_template=qa_prompt,
)

response = hybrid_query_engine.query(
    "Which students need support in Mathematics?"
)

print(response)
```

Hybrid retrieval is useful because:

```text
Vector search = meaning
BM25 = exact words
Hybrid search = both
```

---

## 18. Compare Vector, BM25, and Hybrid Retrieval

```python
query = "Which students need support in Mathematics?"

retriever_list = [
    ("Vector Retriever", vector_retriever),
    ("BM25 Retriever", bm25_retriever),
    ("Hybrid Retriever", hybrid_retriever),
]

for retriever_name, active_retriever in retriever_list:
    print(f"\n================ {retriever_name} ================")

    results = active_retriever.retrieve(query)

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("Score:", result.score)
        print("File:", result.metadata.get("file_name"))
        print(result.text[:700].replace("\n", " "))
```

This comparison helps understand when semantic search, keyword search, or hybrid search performs better.

---

## 19. Reranking

Reranking retrieves multiple candidate chunks and then keeps only the strongest ones.

```python
from llama_index.core.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2",
    top_n=2
)

rerank_query_engine = custom_index.as_query_engine(
    similarity_top_k=6,
    node_postprocessors=[reranker],
    text_qa_template=qa_prompt,
)

response = rerank_query_engine.query(
    "Which students have low attendance and need academic support?"
)

print(response)
```

Flow:

```text
Retrieve top 6 chunks
↓
Rerank
↓
Keep top 2
↓
Generate answer
```

---

## 20. Query Rewriting

Query rewriting improves vague questions before retrieval.

Example:

```text
Original question:
Who is struggling?

Better search query:
Which students have Status: Needs Support, low scores, or low attendance?
```

Code:

```python
def rewrite_question_for_school_data(user_question: str) -> str:
    prompt = f"""
You are improving a search query for a school report dataset.

The dataset contains:
- Grade 5, Grade 6, Grade 7, and Grade 8 reports
- student names
- gender
- subject scores
- attendance
- final average
- status such as Passed or Needs Support

Rewrite the user's question into a clearer search query.
Return only the rewritten query.

User question:
{user_question}
"""

    rewritten = Settings.llm.complete(prompt)
    return str(rewritten).strip()


original_question = "Who is struggling?"

rewritten_question = rewrite_question_for_school_data(original_question)

print("Original question:", original_question)
print("Rewritten question:", rewritten_question)
```

---

## 21. Router Query Engine

The Router Query Engine chooses the best query engine for a question.

In this project, separate query engines are created for Grade 5, Grade 6, Grade 7, and Grade 8.

```python
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

def build_grade_engine(grade_number: int):
    grade_docs = [
        doc for doc in documents
        if f"grade_{grade_number}_report.txt" in doc.metadata.get("file_name", "")
    ]

    grade_index = VectorStoreIndex.from_documents(grade_docs)

    return grade_index.as_query_engine(
        similarity_top_k=2,
        text_qa_template=qa_prompt,
    )

grade_5_engine = build_grade_engine(5)
grade_6_engine = build_grade_engine(6)
grade_7_engine = build_grade_engine(7)
grade_8_engine = build_grade_engine(8)

query_engine_tools = [
    QueryEngineTool.from_defaults(
        query_engine=grade_5_engine,
        description="Useful for questions about Grade 5."
    ),
    QueryEngineTool.from_defaults(
        query_engine=grade_6_engine,
        description="Useful for questions about Grade 6."
    ),
    QueryEngineTool.from_defaults(
        query_engine=grade_7_engine,
        description="Useful for questions about Grade 7."
    ),
    QueryEngineTool.from_defaults(
        query_engine=grade_8_engine,
        description="Useful for questions about Grade 8."
    ),
]

router_query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=query_engine_tools,
)

response = router_query_engine.query(
    "Who is the top student in Grade 8 and what is the final average?"
)

print(response)
```

Expected output:

```text
Betelhem Dawit is the top student in Grade 8 with a final average of 93.25.
```

---

## 22. Sub-Question Query Engine

The Sub-Question Query Engine helps answer questions that require information from multiple documents.

```python
from llama_index.core.query_engine import SubQuestionQueryEngine

sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=False,
)

response = sub_question_engine.query(
    "Compare Grade 6 and Grade 8 performance. Mention top students, class average, strongest subject, and students needing support."
)

print(response)
```

Expected answer should compare:

```text
Grade 6:
Top student: Isaac Daniel
Class average: around 77.2
Strongest subject: Science
Needs support: Natnael Worku and Leul Assefa

Grade 8:
Top student: Betelhem Dawit
Class average: around 84.2
Strongest subject: Science
Needs support: Ermias Kebede
```

---

## 23. Advanced Retrieval Evaluation

This checks whether retrieval returns the correct source file.

```python
retrieval_tests = [
    {
        "question": "Who is the top student in Grade 5?",
        "expected_file": "grade_5_report.txt"
    },
    {
        "question": "Which students need support in Grade 6?",
        "expected_file": "grade_6_report.txt"
    },
    {
        "question": "Who is the top student in Grade 7?",
        "expected_file": "grade_7_report.txt"
    },
    {
        "question": "Who is the top student in Grade 8?",
        "expected_file": "grade_8_report.txt"
    },
]

for test in retrieval_tests:
    results = retriever.retrieve(test["question"])

    retrieved_files = [
        result.metadata.get("file_name", "")
        for result in results
    ]

    passed = test["expected_file"] in retrieved_files

    print("Question:", test["question"])
    print("Expected file:", test["expected_file"])
    print("Retrieved files:", retrieved_files)
    print("Passed:", passed)
    print("-" * 80)
```

Expected output:

```text
Expected file: grade_8_report.txt
Retrieved files: ['grade_8_report.txt', ...]
Passed: True
```

---

## 24. Final Gradio Web App

The final section creates a small web app using Gradio.

The app can:

- Upload TXT/PDF files
- Build a vector index
- Ask questions
- Generate answers
- Show source chunks

```python
import shutil
import gradio as gr
from pathlib import Path

app_query_engine = None

def build_index_from_uploads(files):
    global app_query_engine

    if files is None or len(files) == 0:
        return "Please upload at least one TXT or PDF file."

    upload_dir = Path("uploaded_docs")

    if upload_dir.exists():
        shutil.rmtree(upload_dir)

    upload_dir.mkdir(exist_ok=True)

    for file in files:
        source_path = Path(file.name)
        target_path = upload_dir / source_path.name
        shutil.copy(source_path, target_path)

    documents = SimpleDirectoryReader(str(upload_dir)).load_data()

    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )

    nodes = splitter.get_nodes_from_documents(documents)

    app_index = VectorStoreIndex(nodes)

    app_query_engine = app_index.as_query_engine(
        similarity_top_k=3,
        text_qa_template=qa_prompt
    )

    return f"Index created successfully. Files: {len(files)} | Chunks: {len(nodes)}"


def ask_question(question):
    global app_query_engine

    if app_query_engine is None:
        return "Please upload files and click Build Index first."

    if question is None or not question.strip():
        return "Please enter a question."

    response = app_query_engine.query(question)

    output = "ANSWER\n"
    output += str(response)
    output += "\n\nSOURCE CHUNKS\n"

    for i, source in enumerate(response.source_nodes, start=1):
        text = source.node.text[:700].replace("\n", " ")
        score = source.score
        metadata = source.node.metadata

        output += f"\nSource {i}\n"
        output += f"Score: {score}\n"
        output += f"Metadata: {metadata}\n"
        output += f"Text: {text}\n"
        output += "-" * 70 + "\n"

    return output


with gr.Blocks() as demo:
    gr.Markdown("# LlamaIndex AI Study Assistant")
    gr.Markdown(
        "Upload TXT/PDF files, build a vector index, ask questions, and inspect source chunks."
    )

    files = gr.File(
        label="Upload TXT or PDF files",
        file_count="multiple"
    )

    build_button = gr.Button("Build Index")
    build_status = gr.Textbox(label="Index Status")

    question_box = gr.Textbox(
        label="Ask a question",
        placeholder="Example: Who is the top student in Grade 8?"
    )

    ask_button = gr.Button("Ask")
    answer_box = gr.Textbox(
        label="Answer and Sources",
        lines=22
    )

    build_button.click(
        fn=build_index_from_uploads,
        inputs=files,
        outputs=build_status
    )

    ask_button.click(
        fn=ask_question,
        inputs=question_box,
        outputs=answer_box
    )

demo.launch(share=True)
```

Final app flow:

```text
Upload documents
↓
Build index
↓
Ask question
↓
Generate answer
↓
Show source chunks
```

## Final Architecture

```text
User Documents
↓
SimpleDirectoryReader
↓
SentenceSplitter
↓
HuggingFaceEmbedding
↓
VectorStoreIndex
↓
Retriever
↓
OpenRouter LLM
↓
Answer + Sources
↓
Gradio Web App
```

## Skills Demonstrated

This project demonstrates practical experience with:

- Retrieval-Augmented Generation
- LlamaIndex
- Document ingestion
- Chunking strategies
- Embedding models
- Vector indexing
- Semantic search
- Keyword search with BM25
- Hybrid retrieval
- Reranking
- Prompt design
- Source-grounded answering
- Query rewriting
- Router-based query engines
- Sub-question decomposition
- Basic RAG evaluation
- Gradio app development
- Kaggle-based AI prototyping

## Key Takeaway

A useful RAG system is not only about sending documents to an LLM.

The important work is controlling the full pipeline:

```text
Load
↓
Chunk
↓
Embed
↓
Index
↓
Retrieve
↓
Rerank
↓
Prompt
↓
Answer
↓
Inspect
↓
Evaluate
↓
Deploy
```

This notebook shows that complete workflow in a clear and practical way.
