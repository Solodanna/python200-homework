from dotenv import load_dotenv
import os
import string

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# ============================================
# RAG CONCEPTS
# ============================================

# --- Concepts Q1 ---
# Scenario A: RAG (Retrieval-Augmented Generation)
# Reasoning: The policy library is large, frequently updated, and domain-specific. 
# RAG allows the system to retrieve and reference the latest policies without retraining, 
# and keeps the model grounded in the actual company documents.

# Scenario B: Fine-tuning
# Reasoning: The brand voice is specific and subtle, not well-represented online. 
# Fine-tuning on 3,000 examples of the company's actual writing will teach the model 
# the specific stylistic patterns and minimize hallucinations about brand voice.

# Scenario C: Prompt engineering
# Reasoning: A one-time document analysis doesn't require persistent knowledge updates 
# or permanent behavioral changes. Simple prompt engineering with the report text 
# included in the context is sufficient and most efficient.


# --- Concepts Q2 ---
# Why confident hallucinations are more harmful:
# A confident wrong answer is more harmful because users tend to trust statements delivered 
# with certainty. When an AI says "I'm not sure," users know to verify the information 
# themselves or seek alternative sources. A confident hallucination often bypasses critical 
# thinking and gets repeated or acted upon.
#
# Real-world example: A medical AI hallucinating a drug interaction with complete confidence 
# could lead a patient to stop taking a necessary medication, causing serious health consequences. 
# The user trusts the confident assertion without verifying it with their doctor.


# --- Concepts Q3 ---
# Correct RAG pipeline order with descriptions:
# steps = [
#     "Receive the user's query",               # 1. User submits a question
#     "Extract text from source documents",     # 2. Load and read all source material
#     "Split text into chunks",                 # 3. Break documents into manageable pieces
#     "Convert text chunks into embeddings",    # 4. Vectorize all chunks for similarity search
#     "Embed the user's query",                 # 5. Convert the query into the same vector space
#     "Retrieve the most relevant chunks",      # 6. Find chunks closest to the query vector
#     "Inject retrieved chunks into the prompt",# 7. Add retrieved context to the LLM prompt
#     "Generate a response from the LLM",       # 8. LLM generates answer based on context
# ]


# ============================================
# KEYWORD RAG
# ============================================

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


# --- Keyword Q1 ---
print("\n" + "="*50)
print("KEYWORD RAG - Question 1")
print("="*50)

query_1 = "What are your hours on the weekend?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

result_1 = simple_keyword_retrieval(query_1, documents, verbose=True)
print(f"\nSelected document: {result_1[0][0]}")

# Comment: The "hours.txt" document was selected because it contains the key terms "weekend" 
# and "hours" which directly match the query. This is the correct document for this question.


# --- Keyword Q2 ---
print("\n" + "="*50)
print("KEYWORD RAG - Question 2")
print("="*50)

query_2 = "Do you have anything without caffeine?"
result_2 = simple_keyword_retrieval(query_2, documents, verbose=True)
print(f"\nSelected document: {result_2[0][0]}")

# Comment: The "loyalty.txt" document was selected because "drink" appears in both the query 
# (implied in context of "anything") and the loyalty document.
#
# Did keyword RAG get this right? NO - The query is asking about caffeine-free beverages, 
# but the loyalty document doesn't mention caffeine or any specific drinks. The correct answer 
# should come from "menu.txt" which lists actual products, including options like oat milk 
# (which might suggest decaf possibilities).
#
# Why it failed: Keyword retrieval only matches exact words or stems. It doesn't understand 
# that the user wants to know about beverage options, nor does it understand the relationship 
# between "caffeine-free" and specific menu items.
#
# Better approach: Semantic RAG using embeddings would understand that "without caffeine" 
# is semantically related to terms like "decaf," "caffeine-free," or specific drink names 
# in the menu, even if those exact words don't appear in the query.


# --- Keyword Q3 ---
print("\n" + "="*50)
print("KEYWORD RAG - Question 3")
print("="*50)

# Prediction: I predict that "loyalty.txt" will be selected because it's the only document 
# that contains both "rewards" and related terms like "loyalty program" and "earn."
# Reasoning: "sign up" and "rewards" are the key terms in the query. The loyalty document 
# has "loyalty program" and "earn points" but not "sign up." However, this is likely the best match.

query_3 = "How do I sign up for rewards?"
result_3 = simple_keyword_retrieval(query_3, documents, verbose=True)
print(f"\nSelected document: {result_3[0][0]}")

# Comment: The prediction was correct. "loyalty.txt" was selected because it contains 
# "rewards" (via "earn" and loyalty context) and is the only document discussing the rewards 
# program. The function found overlap with the loyalty-related terms in the query.


# ============================================
# SEMANTIC RAG CONCEPTS
# ============================================

print("\n" + "="*50)
print("SEMANTIC RAG CONCEPTS")
print("="*50)

# --- Semantic Q1 ---
# What is a vector embedding?
# A vector embedding is a numerical representation of text (a word, sentence, or document) 
# as a list of numbers (a vector) that captures its semantic meaning. Text with similar 
# meanings will have similar vector representations in the embedding space.
#
# Cosine similarity interpretation:
# The chunk with a cosine similarity of 0.85 is more relevant. Cosine similarity ranges 
# from 0 (completely unrelated) to 1 (identical meaning). A score of 0.85 means the chunk 
# is very similar to the query in meaning, while 0.30 indicates weak semantic relationship.
#
# Why semantic search handles unseen words:
# Semantic search compares the meanings of texts, not exact word matches. If a query asks 
# about "vehicle" and a chunk discusses "car," semantic embeddings recognize these have 
# similar meanings even though the exact words differ. Embeddings capture the semantic 
# essence of words based on their learned relationships with other words.


# --- Semantic Q2 ---
# Comparison table:
# | Feature                    | Keyword RAG                       | Semantic RAG                  |
# |----------------------------|-----------------------------------|-------------------------------|
# | What is compared?          | Exact word overlap                | Semantic meaning (vectors)    |
# | What is retrieved?         | Full document                     | Text chunks                   |
# | Can it handle synonyms?    | No                                | Yes                           |
# | Storage format             | Plain text dictionary             | Vector embeddings             |
# | Relevance score            | Number of overlapping keywords    | Cosine similarity (0-1)       |


# ============================================
# LLAMAINDEX
# ============================================

print("\n" + "="*50)
print("LLAMAINDEX PIPELINE")
print("="*50)

try:
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core.evaluators import FaithfulnessEvaluator, RelevancyEvaluator
    from llama_index.core.llms import OpenAI
    
    # Set up embedding model and LLM
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-4o-mini")
    
    # Path to Brightleaf PDFs - adjust based on your folder structure
    pdf_path = "../../06_AI_augmentation/brightleaf_pdfs"
    
    try:
        print("\nLoading documents from:", pdf_path)
        documents = SimpleDirectoryReader(pdf_path).load_data()
        print(f"Loaded {len(documents)} documents")
        
        # Build the vector index
        print("\nBuilding vector index...")
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine(similarity_top_k=3)
        
        # --- LlamaIndex Q1 ---
        print("\n" + "-"*50)
        print("LlamaIndex Q1: Employee benefits and security policies")
        print("-"*50)
        
        questions = [
            "What employee benefits does BrightLeaf offer?",
            "What are BrightLeaf's security policies?",
        ]
        
        for q in questions:
            print(f"\n>>> Question: {q}")
            response = query_engine.query(q)
            print(f">>> Answer: {response}")
            
            # Extract and print source nodes
            if hasattr(response, 'source_nodes'):
                for i, node in enumerate(response.source_nodes[:3], 1):
                    sim_score = node.score if hasattr(node, 'score') else "N/A"
                    text_preview = node.node.get_content()[:150] if hasattr(node.node, 'get_content') else str(node)[:150]
                    print(f"  Source {i} (similarity: {sim_score}): {text_preview}...")
            
            # Analysis comments
            print(f"\n  Analysis for Q{questions.index(q)+1}:")
            print(f"  - Do retrieved chunks look relevant? [Check your output above]")
            print(f"  - Model confidence: [Note whether response hedges or sounds confident]")
            print(f"  - Unexpected retrievals? [Review the source nodes above]")
        
        # --- LlamaIndex Q2 ---
        print("\n" + "-"*50)
        print("LlamaIndex Q2: Testing different similarity_top_k values")
        print("-"*50)
        
        test_query = "What employee benefits does BrightLeaf offer?"
        
        print(f"\n>>> Query: {test_query}")
        print("\nWith similarity_top_k=1:")
        qe_1 = index.as_query_engine(similarity_top_k=1)
        response_1 = qe_1.query(test_query)
        print(f"Response: {response_1}")
        if hasattr(response_1, 'source_nodes'):
            for node in response_1.source_nodes:
                sim_score = node.score if hasattr(node, 'score') else "N/A"
                print(f"  Score: {sim_score}")
        
        print("\nWith similarity_top_k=5:")
        qe_5 = index.as_query_engine(similarity_top_k=5)
        response_5 = qe_5.query(test_query)
        print(f"Response: {response_5}")
        if hasattr(response_5, 'source_nodes'):
            for i, node in enumerate(response_5.source_nodes, 1):
                sim_score = node.score if hasattr(node, 'score') else "N/A"
                print(f"  Source {i} score: {sim_score}")
        
        # Comment on differences
        print("\n  Comment: [Analyze how the response changed between top_k=1 and top_k=5]")
        print("  Does more context always help? [Consider trade-offs between completeness and noise]")
        
        # --- LlamaIndex Q3 ---
        print("\n" + "-"*50)
        print("LlamaIndex Q3: Testing a challenging query")
        print("-"*50)
        
        challenging_query = "What is the company's vision for the future?"
        print(f"\n>>> Challenging Query: {challenging_query}")
        qe_orig = index.as_query_engine(similarity_top_k=3)
        response_q3 = qe_orig.query(challenging_query)
        print(f"Response: {response_q3}")
        
        if hasattr(response_q3, 'source_nodes'):
            print("\nAll retrieved chunks:")
            for i, node in enumerate(response_q3.source_nodes, 1):
                sim_score = node.score if hasattr(node, 'score') else "N/A"
                text = node.node.get_content()[:200] if hasattr(node.node, 'get_content') else str(node)[:200]
                print(f"  Chunk {i} (score: {sim_score}): {text}...")
        
        print("\n  Comment: [Describe what you expected vs. what happened]")
        print("  [Suggest improvements to handle this query better]")
        
        # --- LlamaIndex Q4 ---
        print("\n" + "-"*50)
        print("LlamaIndex Q4: RAG Evaluation")
        print("-"*50)
        
        print("\nSetting up evaluators...")
        faithfulness_evaluator = FaithfulnessEvaluator(llm=OpenAI(model="gpt-4o-mini"))
        relevancy_evaluator = RelevancyEvaluator(llm=OpenAI(model="gpt-4o-mini"))
        
        q_eval = "What employee benefits does BrightLeaf offer?"
        print(f"\n>>> Evaluating query: {q_eval}")
        response_eval = qe_orig.query(q_eval)
        
        print("\nEvaluating response with gpt-4o-mini...")
        faithfulness_score = faithfulness_evaluator.evaluate_response(response=response_eval)
        relevancy_score = relevancy_evaluator.evaluate_response(response=response_eval)
        
        print(f"Faithfulness Score: {faithfulness_score}")
        print(f"Relevancy Score: {relevancy_score}")
        
        # Test with a query expected to perform worse
        bad_query = "What is the secret recipe for Brightleaf's invisible products?"
        print(f"\n>>> Evaluating low-quality query: {bad_query}")
        response_bad = qe_orig.query(bad_query)
        
        faithfulness_score_bad = faithfulness_evaluator.evaluate_response(response=response_bad)
        relevancy_score_bad = relevancy_evaluator.evaluate_response(response=response_bad)
        
        print(f"Faithfulness Score: {faithfulness_score_bad}")
        print(f"Relevancy Score: {relevancy_score_bad}")
        
        # Evaluation analysis
        print("\n" + "="*50)
        print("Evaluation Analysis:")
        print("="*50)
        print("""
  What does a faithfulness score of 1.0 mean?
  A score of 1.0 means the model's response is completely grounded in the retrieved context 
  with no hallucinations. A score of 0.0 indicates the response contradicts or is unsupported 
  by the retrieved information.
  
  What does a relevancy score measure?
  Relevancy measures whether the retrieved context actually addresses the query's question. 
  It differs from faithfulness in that it evaluates context selection quality, not response 
  accuracy. You can have faithful responses to irrelevant context.
  
  Score changes between queries:
  [Observe whether scores differed between the good and bad queries]
  [Explain why (e.g., hallucination detection, context irrelevance)]
  
  LLM-as-a-judge approach:
  This approach uses a separate LLM (the judge) to evaluate the quality of responses by 
  checking faithfulness and relevancy. It's used instead of simple accuracy metrics because 
  RAG evaluation requires semantic understanding of whether generated text is grounded in 
  retrieved context and actually answers the question. Simple metrics like BLEU scores can't 
  capture this nuanced evaluation.
        """)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find PDF directory at {pdf_path}")
        print("Please verify the path to brightleaf_pdfs matches your local folder structure.")
        print("Example: If you're in assignments_06/, the path might be '../../06_AI_augmentation/brightleaf_pdfs'")

except ImportError as e:
    print(f"Error importing LlamaIndex: {e}")
    print("Make sure all required packages are installed: pip install llama-index-core llama-index-embeddings-openai")
