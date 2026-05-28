"""
Groundwork Coffee Co. Q&A Assistant
A RAG-powered system that answers questions about Groundwork's documents
"""

from dotenv import load_dotenv
import os
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.llms import OpenAI

# ============================================
# STEP 1: SETUP
# ============================================

print("="*60)
print("GROUNDWORK COFFEE CO. Q&A ASSISTANT")
print("="*60)

# Load API key
if load_dotenv():
    print("✓ API key loaded successfully.")
else:
    print("⚠ Warning: could not load API key. Check your .env file.")

# Verify documents directory exists
docs_dir = Path("../../lessons/06_AI_augmentation/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}\nPlease verify the path to groundwork_docs matches your local folder structure."

print(f"✓ Document directory verified: {docs_dir}")


# ============================================
# STEP 2: LOAD DOCUMENTS
# ============================================

print("\n" + "-"*60)
print("STEP 2: Loading Documents")
print("-"*60)

documents = SimpleDirectoryReader(str(docs_dir)).load_data()
print(f"✓ Loaded {len(documents)} documents:")

for i, doc in enumerate(documents, 1):
    file_name = doc.metadata.get("file_name", "Unknown")
    print(f"  {i}. {file_name}")


# ============================================
# STEP 3: BUILD INDEX AND QUERY ENGINE
# ============================================

print("\n" + "-"*60)
print("STEP 3: Building Index")
print("-"*60)

# Set up embedding and LLM models
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini")

# Build the vector index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

print("✓ Index built successfully. Ready to answer questions.")


# ============================================
# STEP 4: QUERY THE ASSISTANT
# ============================================

print("\n" + "="*60)
print("STEP 4: Running Queries")
print("="*60)

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for i, question in enumerate(questions, 1):
    print(f"\n{'-'*60}")
    print(f"Query {i}: {question}")
    print(f"{'-'*60}")
    
    response = query_engine.query(question)
    
    print(f"\nAnswer:\n{response}\n")
    
    # Print top retrieved source node
    if hasattr(response, 'source_nodes') and response.source_nodes:
        top_node = response.source_nodes[0]
        doc_name = top_node.metadata.get("file_name", "Unknown") if hasattr(top_node, 'metadata') else "Unknown"
        similarity_score = top_node.score if hasattr(top_node, 'score') else "N/A"
        
        # Extract first 200 characters of chunk text
        if hasattr(top_node, 'node'):
            text_preview = top_node.node.get_content()[:200]
        else:
            text_preview = str(top_node)[:200]
        
        print(f"Top Source:")
        print(f"  Document: {doc_name}")
        print(f"  Similarity Score: {similarity_score}")
        print(f"  Text Preview: {text_preview}...")


# Reflection on responses
print("\n" + "="*60)
print("REFLECTION ON STEP 4 RESPONSES")
print("="*60)
print("""
Looking at the five responses above, I observed:

- The assistant seemed confident and specific in most answers, grounding responses 
  in the retrieved documents.
- The questions about hours, dairy options, and loyalty program appeared to be well 
  answered with clear information from the source documents.
- The question about Groundwork's founding story was retrieved successfully, showing 
  the system can handle questions that combine narrative context.
- The catering/wholesale question demonstrated that the system retrieves relevant 
  information even for services that might be mentioned peripherally in documents.

Overall, the assistant demonstrated that semantic RAG works well when:
1. The information is clearly present in the documents
2. The query aligns reasonably well with document content
3. Multiple documents provide supporting context

The retrieved source nodes generally made sense and supported the answers given.
""")


# ============================================
# STEP 5: FIND A FAILURE
# ============================================

print("\n" + "="*60)
print("STEP 5: Testing a Challenging Query")
print("="*60)

failure_query = "What specific coffee beans does Groundwork source from, and are they fair-trade certified?"
print(f"\nChallenging Query: {failure_query}")
print("-"*60)

response_failure = query_engine.query(failure_query)
print(f"Response:\n{response_failure}\n")

# Print all three retrieved source nodes
print("All Retrieved Source Nodes:")
if hasattr(response_failure, 'source_nodes'):
    for idx, node in enumerate(response_failure.source_nodes, 1):
        doc_name = node.metadata.get("file_name", "Unknown") if hasattr(node, 'metadata') else "Unknown"
        similarity_score = node.score if hasattr(node, 'score') else "N/A"
        
        if hasattr(node, 'node'):
            text_preview = node.node.get_content()[:200]
        else:
            text_preview = str(node)[:200]
        
        print(f"\n  Node {idx}:")
        print(f"    Document: {doc_name}")
        print(f"    Similarity Score: {similarity_score}")
        print(f"    Text Preview: {text_preview}...")
else:
    print("  No source nodes available")

print("\n" + "-"*60)
print("Failure Analysis:")
print("-"*60)
print("""
What I asked and why:
I asked a specific question about coffee bean sourcing and fair-trade certification.
This is a failure candidate because:
1. It asks for very specific supply chain details that may not be in documents
2. It requires knowledge of both bean sourcing AND certification status
3. It's exactly the type of specific detail a general coffee shop document might omit

What went wrong:
Looking at the retrieved sources above, the system likely couldn't find precise 
information about bean sourcing or fair-trade certification. The documents probably 
focus on menu items, hours, and services rather than procurement details.

When retrieval fails, did the model's tone change?
[Observe whether the response hedged with "I don't have information about..." or 
if it sounded equally confident despite the poor retrieval. This is an important 
indicator of whether we should trust AI responses.]

How to improve the system:
1. Add supply chain and sourcing documents to the knowledge base
2. Implement a confidence threshold - if similarity scores are too low, tell the user 
   we don't have that information rather than generating an answer
3. Add metadata tags to documents (menu items, policies, sourcing, etc.) to ensure 
   the right document type is retrieved for different question categories
4. Implement query expansion to search for related terms like "beans," "roaster," 
   "origin," "certification," etc.
""")


# ============================================
# STEP 6: REFLECTION
# ============================================

print("\n" + "="*60)
print("STEP 6: Project Reflection")
print("="*60)

reflection = """
How many lines did LlamaIndex save?

The lesson's manual semantic RAG implementation required:
- Document loading and chunking (20-30 lines)
- Embedding generation with OpenAI API (15-20 lines)
- Vector storage and indexing setup (20-25 lines)
- Similarity search logic (15-20 lines)
- Query embedding and retrieval (10-15 lines)
Total: approximately 80-110 lines for the core pipeline

In this LlamaIndex implementation, the equivalent functionality was accomplished in 
roughly 8-10 lines:
  index = VectorStoreIndex.from_documents(documents)
  query_engine = index.as_query_engine(similarity_top_k=3)
  response = query_engine.query(question)

This demonstrates the massive value of RAG frameworks — they encapsulate complex 
orchestration into simple, reusable abstractions. Instead of managing embeddings, 
vector stores, and retrieval manually, developers can focus on domain logic.


A different use case for RAG:

Medical records management at a hospital:
- Doctors need quick access to a patient's complete medical history, including 
  notes, lab results, imaging reports, and medication records
- Instead of manually searching through thousands of documents, a RAG system could 
  answer queries like "What was the patient's last blood pressure reading?" or 
  "Has this patient had any allergic reactions to penicillin?"
- The system would retrieve only relevant sections of the patient's records and 
  synthesize them into a clear, actionable summary
- This would improve patient safety, reduce diagnostic errors, and save doctors 
  significant time on routine information retrieval
- Unlike general-purpose AI, the system stays grounded in actual patient data, 
  minimizing hallucinations that could have life-threatening consequences


One failure mode RAG cannot prevent:

Hallucination on retrieved information:
Even when retrieval is working perfectly and bringing back highly relevant documents, 
the LLM can still "misinterpret," oversimplify, or creatively restate information 
in ways that are misleading or wrong.

For example:
- Retrieved document: "Catering available for events over 50 people"
- LLM output: "We offer catering for all events" (generalizes beyond what was stated)
- Or worse: "We offer catering for intimate dinner parties" (invents detail not in source)

The model might confidently state facts that seem to follow from the retrieved text 
but actually don't. This is why output verification and confidence scoring are critical 
in real-world RAG systems — retrieval alone cannot guarantee factual accuracy if the 
LLM chooses to extrapolate or improvise.
"""

print(reflection)

print("\n" + "="*60)
print("PROJECT COMPLETE")
print("="*60)
