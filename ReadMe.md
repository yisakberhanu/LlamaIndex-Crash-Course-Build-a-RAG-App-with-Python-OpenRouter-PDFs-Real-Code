I built a complete LlamaIndex RAG crash course in Kaggle using Python, OpenRouter, HuggingFace embeddings, and Gradio.

The goal was to build a real document Q&A app step by step — not just explain RAG theory.

The project:

AI Study Assistant Upload TXT/PDF → Build Index → Ask Questions → Get Answer → Inspect Sources

Full pipeline:

Documents → Chunks → Embeddings → Vector Index → Retriever → LLM → Answer

Here is the full breakdown with code.


Install packages

We use LlamaIndex, OpenRouter, HuggingFace embeddings, PDF support, and Gradio.

!pip install -q llama-index llama-index-llms-openai \
llama-index-embeddings-huggingface sentence-transformers pypdf gradio

For advanced retrieval:

!pip install -q llama-index-retrievers-bm25 rank-bm25 sentence-transformers


Set OpenRouter API key

In Kaggle, I use Secrets.

Secret name:

OPENROUTER_API_KEY

Code:

import os

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    os.environ["OPENROUTER_API_KEY"] = user_secrets.get_secret("OPENROUTER_API_KEY")
    print("OpenRouter API key loaded.")
except Exception:
    print("Could not load Kaggle Secret.")


Configure LlamaIndex

I use:

OpenRouter for the LLM

HuggingFace for embeddings

Why?

OpenRouter handles answer generation. HuggingFace embeddings avoid embedding quota issues.

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


Create fake school documents

To make the tutorial easy to understand, I created fake school reports for Grade 5, 6, 7, and 8.

Each document includes:

Student name

Gender

Subject scores

Attendance

Final average

Status

Grade summary

Example:

from pathlib import Path

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

(data_dir / "grade_8_report.txt").write_text("""
School: Bright Future Academy
Academic Year: 2026
Grade: Grade 8
Class Teacher: Mr. Birhanu Desta

Student Records:
1. Amen Tesfaye | Gender: Male | English: 86 | Mathematics: 89 | Science: 88 | Social Studies: 85 | Attendance: 96% | Final Average: 87.00 | Status: Passed
2. Betelhem Dawit | Gender: Female | English: 92 | Mathematics: 94 | Science: 96 | Social Studies: 91 | Attendance: 99% | Final Average: 93.25 | Status: Passed
3. Caleb Yared | Gender: Male | English: 79 | Mathematics: 82 | Science: 84 | Social Studies: 80 | Attendance: 93% | Final Average: 81.25 | Status: Passed
4. Danait Alemayehu | Gender: Female | English: 88 | Mathematics: 90 | Science: 91 | Social Studies: 89 | Attendance: 97% | Final Average: 89.50 | Status: Passed
5. Ermias Kebede | Gender: Male | English: 69 | Mathematics: 64 | Science: 70 | Social Studies: 72 | Attendance: 84% | Final Average: 68.75 | Status: Needs Support
6. Lidiya Samuel | Gender: Female | English: 84 | Mathematics: 87 | Science: 85 | Social Studies: 86 | Attendance: 95% | Final Average: 85.50 | Status: Passed

Grade 8 Summary:
Betelhem Dawit is the top student in Grade 8 with a final average of 93.25.
Ermias Kebede needs additional support, especially in Mathematics.
The class average is around 84.2.
The strongest subject for Grade 8 is Science.
Grade 8 has the highest class average among Grade 5, Grade 6, Grade 7, and Grade 8.
""", encoding="utf-8")

Example questions:

Who is the top student in Grade 8?
Which students need support in Mathematics?
Which grade has the highest class average?
How many female students are in Grade 8?
What is the school lunch menu?

The last question is useful because the answer is not in the documents.


Load documents

LlamaIndex loads files using SimpleDirectoryReader.

from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()

print(f"Number of loaded documents: {len(documents)}")

for doc in documents:
    print(doc.metadata.get("file_name"))

Expected output:

Number of loaded documents: 4
grade_5_report.txt
grade_6_report.txt
grade_7_report.txt
grade_8_report.txt

This turns raw files into LlamaIndex Document objects.


Build vector index

Now we create a searchable vector index.

from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)

print("Vector index created.")

What happens inside:

documents
↓
chunks
↓
embeddings
↓
vector index


Ask the first question

query_engine = index.as_query_engine(similarity_top_k=3)

response = query_engine.query("Who is the top student in Grade 8?")

print(response)

Expected output:

Betelhem Dawit is the top student in Grade 8 with a final average of 93.25.

This is the first complete RAG loop:

question → retrieve context → LLM answer


Inspect retrieval only

Before trusting the answer, I inspect what the retriever finds.

retriever = index.as_retriever(similarity_top_k=3)

results = retriever.retrieve("Which students need support in Grade 7?")

for i, node in enumerate(results, start=1):
    print(f"Result {i}")
    print("Score:", node.score)
    print("File:", node.metadata.get("file_name"))
    print(node.text[:500])
    print("-" * 80)

Expected file:

grade_7_report.txt

Expected content includes:

Yosef Fikru
Tinsae Solomon
Needs Support
Mathematics

This helps check if retrieval is working before the LLM generates the final answer.


Inspect source nodes

Source nodes show the exact chunks used to answer.

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

Expected source:

grade_8_report.txt

This makes the answer easier to verify.


Chunking and chunk overlap

Chunking means splitting long documents into smaller parts.

from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=256,
    chunk_overlap=30
)

nodes = splitter.get_nodes_from_documents(documents)

print(f"Number of chunks/nodes created: {len(nodes)}")

Expected output:

Number of chunks/nodes created: 10

To see overlap clearly:

for i in range(len(nodes) - 1):
    print(f"\n========== Node {i+1} → Node {i+2} ==========")

    print("\nEND OF CURRENT NODE:")
    print(nodes[i].text[-250:])

    print("\nSTART OF NEXT NODE:")
    print(nodes[i + 1].text[:250])

Why overlap matters:

If important information is split between two chunks, overlap helps preserve context.


Build index from custom chunks

custom_index = VectorStoreIndex(nodes)

custom_query_engine = custom_index.as_query_engine(
    similarity_top_k=3
)

response = custom_query_engine.query(
    "Which students need support in Mathematics?"
)

print(response)

Expected output should mention students like:

Yonas Mengistu
Dawit Kebede
Yosef Fikru
Tinsae Solomon
Ermias Kebede

Custom chunking gives better control over retrieval.


Custom prompt

A custom prompt helps stop the model from inventing unsupported answers.

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

Expected output:

I don't know based on the provided documents.

This is important because the lunch menu is not in the school documents.


PDF question answering

Same RAG pipeline, different file type.

from pathlib import Path

pdf_dir = Path("pdf_data")
pdf_dir.mkdir(exist_ok=True)

print("Upload PDF files into:", pdf_dir.resolve())

After uploading PDFs:

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

Expected output:

Loaded PDF document objects.
Summary:
...


Save and reload index

Creating embeddings every time is wasteful.

Save index:

index.storage_context.persist(persist_dir="storage")

print("Index saved.")

Reload index:

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

Expected output:

Betelhem Dawit is the top student in Grade 8 with a final average of 93.25.


Chat engine

The chat engine supports follow-up questions.

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

Expected output:

Reply 1:
Betelhem Dawit is the top student in Grade 8.

Reply 2:
Her final average is 93.25.

The second question depends on the first one.


Simple evaluation

I added a small evaluation loop.

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

Expected output:

Question: Who is the top student in Grade 7?
Expected keyword: Sara Abebe
Passed: True

Question: Which grade has the highest class average?
Expected keyword: Grade 8
Passed: True

This is a simple way to start testing a RAG system.


Tools and agents

A tool is a Python function the agent can use.

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

Expected output:

7006652

Difference:

Query engine = answers from documents
Agent = can use tools


Hybrid search

Hybrid search combines vector search and BM25 keyword search.

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

Why it matters:

Vector search = meaning
BM25 = exact words
Hybrid = both


Compare vector, BM25, and hybrid retrieval

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

This helps compare which retriever finds better chunks.


Reranking

Reranking retrieves more chunks first, then keeps the best ones.

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

Flow:

Retrieve top 6 chunks
↓
Rerank
↓
Keep top 2
↓
Generate answer


Query rewriting

Query rewriting makes vague questions clearer.

def rewrite_question_for_school_data(user_question: str) -> str:
    prompt = f'''
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
'''

    rewritten = Settings.llm.complete(prompt)
    return str(rewritten).strip()


original_question = "Who is struggling?"

rewritten_question = rewrite_question_for_school_data(original_question)

print("Original question:", original_question)
print("Rewritten question:", rewritten_question)

Expected rewritten query:

Which students have Status: Needs Support, low scores, or low attendance?

This connects natural language to document language.


Router Query Engine

The router chooses the best query engine.

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

Expected output:

Betelhem Dawit is the top student in Grade 8 with a final average of 93.25.


Sub-Question Query Engine

This helps answer complex questions across multiple documents.

from llama_index.core.query_engine import SubQuestionQueryEngine

sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=False,
)

response = sub_question_engine.query(
    "Compare Grade 6 and Grade 8 performance. Mention top students, class average, strongest subject, and students needing support."
)

print(response)

Expected answer should compare:

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


Advanced retrieval evaluation

This checks if retrieval returns the correct file.

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

Expected output:

Expected file: grade_8_report.txt
Retrieved files: ['grade_8_report.txt', ...]
Passed: True


Final Gradio web app

The final app uploads documents, builds an index, answers questions, and shows source chunks.

import shutil
import gradio as gr

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

Final app flow:

Upload documents
↓
Build index
↓
Ask question
↓
Generate answer
↓
Show source chunks


Full architecture:

User documents
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

This project covers the full LlamaIndex workflow:

RAG basics

Document loading

Chunking

Embeddings

Vector search

Source inspection

Custom prompts

PDF Q&A

Index persistence

Chat engine

Simple evaluation

Tools and agents

Hybrid search

Reranking

Query rewriting

Router Query Engine

Sub-Question Query Engine

Final web app

Main idea:

A good RAG app is not just “send documents to an LLM.”

A good RAG app controls the full pipeline:

Load → Chunk → Embed → Retrieve → Rerank → Prompt → Answer → Inspect → Evaluate → Deploy
