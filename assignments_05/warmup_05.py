# warmup_05.py

import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# --- Completions API ---

# API Q1
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print("API Q1 - Response text:")
print(response.choices[0].message.content)
print("\nAPI Q1 - Model that responded:", response.model)
print("API Q1 - Total tokens used:", response.usage.total_tokens)

# API Q2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

print("\nAPI Q2 - Responses at different temperatures:")
for temp in temperatures:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    print(f"\n  Temperature={temp}:")
    print(" ", resp.choices[0].message.content)

# At temperature=0 the output is very consistent and nearly identical each run.
# At 0.7 there is moderate variation — still coherent but more creative.
# At 1.5 the output is the most varied and sometimes unusual or unexpected.
# For a consistent, reproducible output I would use temperature=0.

# API Q3
print("\nAPI Q3 - Three completions in one call (n=3, temperature=1.0):")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)
for i, choice in enumerate(response.choices, start=1):
    print(f"  Choice {i}: {choice.message.content}")

# API Q4
print("\nAPI Q4 - max_tokens=15 on a long-answer prompt:")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)
print(response.choices[0].message.content)

# The response is cut off mid-sentence because the model stopped after 15 tokens.
# max_tokens is useful in real applications to control cost, prevent runaway outputs,
# and enforce concise answers when a short reply is all that is needed.


# --- System Messages and Personas ---

# System Q1
print("\nSystem Q1 - Python tutor persona:")
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(response.choices[0].message.content)

print("\nSystem Q1 - Grumpy senior engineer persona:")
messages_alt = [
    {"role": "system", "content": "You are a grumpy senior software engineer who has seen it all. You answer questions correctly but with obvious impatience and dry sarcasm."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response_alt = client.chat.completions.create(model="gpt-4o-mini", messages=messages_alt)
print(response_alt.choices[0].message.content)

# The tone changed completely. The tutor persona was warm and simple; the grumpy engineer
# persona was curt and sarcastic. The factual content was similar, but the delivery and
# emotional register were entirely different — showing how much the system message shapes
# the model's behavior.

# System Q2
print("\nSystem Q2 - Stateless multi-turn conversation:")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(response.choices[0].message.content)

# The model knows Jordan's name because we passed the full conversation history in the
# messages list. The API itself is stateless — it stores nothing between calls — but by
# sending prior turns as context the model can "remember" anything that appeared earlier
# in the same request.


# --- Prompt Engineering ---

# Prompt Q1 - Zero-Shot
print("\nPrompt Q1 - Zero-Shot Sentiment:")
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, start=1):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": (
                "Classify the sentiment of the following review as positive, negative, or mixed. "
                "Reply with just the single word.\n\n"
                f"Review: \"{review}\""
            )}
        ]
    )
    print(f"  Review {i}: {resp.choices[0].message.content.strip()}")

# Prompt Q2 - One-Shot
print("\nPrompt Q2 - One-Shot Sentiment:")
one_shot_prefix = (
    "Classify the sentiment of each review as positive, negative, or mixed.\n\n"
    "Example:\n"
    "Review: \"Fast shipping but the item arrived damaged.\"\n"
    "Sentiment: mixed\n\n"
)
for i, review in enumerate(reviews, start=1):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": (
                one_shot_prefix +
                f"Review: \"{review}\"\n"
                "Sentiment:"
            )}
        ]
    )
    print(f"  Review {i}: {resp.choices[0].message.content.strip()}")

# Adding one example made the output format more consistent — the model reliably
# returned just the sentiment word rather than a full sentence. It helped anchor
# the expected response structure.

# Prompt Q3 - Few-Shot
print("\nPrompt Q3 - Few-Shot Sentiment:")
few_shot_prefix = (
    "Classify the sentiment of each review as positive, negative, or mixed.\n\n"
    "Example:\n"
    "Review: \"The team went above and beyond to help us migrate our data.\"\n"
    "Sentiment: positive\n\n"
    "Example:\n"
    "Review: \"The product stopped working after two days and customer service ignored us.\"\n"
    "Sentiment: negative\n\n"
    "Example:\n"
    "Review: \"Fast shipping but the item arrived damaged.\"\n"
    "Sentiment: mixed\n\n"
)
for i, review in enumerate(reviews, start=1):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": (
                few_shot_prefix +
                f"Review: \"{review}\"\n"
                "Sentiment:"
            )}
        ]
    )
    print(f"  Review {i}: {resp.choices[0].message.content.strip()}")

# Comparison of zero-shot, one-shot, and few-shot:
# Zero-shot — convenient and requires no examples, but output format can vary.
#   Use when the task is simple and consistency is not critical.
# One-shot — a single example locks in the output format with minimal extra tokens.
#   Use when you need consistent formatting and have one clear example.
# Few-shot — multiple examples across different categories give the model the clearest
#   signal and produce the most reliable, consistent results.
#   Use for nuanced classification tasks or when accuracy and format consistency matter most.

# Prompt Q4 - Chain of Thought
print("\nPrompt Q4 - Chain-of-Thought Reasoning:")
cot_prompt = (
    "Solve the following problem. Show your reasoning step by step, "
    "then clearly label the final answer.\n\n"
    "A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later "
    "takes a new job that pays $7,500 more per year than her post-raise salary. "
    "What is her final annual salary?"
)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": cot_prompt}]
)
print(response.choices[0].message.content)

# Asking the model to reason step by step tends to improve accuracy because it forces
# the model to decompose the problem into smaller, verifiable sub-steps rather than
# jumping to an answer. Each intermediate result is checked against the previous one,
# which reduces compounding arithmetic or logical errors.

# Prompt Q5 - Structured Output
print("\nPrompt Q5 - Structured JSON Output:")
review = (
    "I've been using this tool for three months. It handles large datasets well, "
    "but the UI is clunky and the export options are limited."
)
structured_prompt = (
    "Analyze the review below and return the result only as valid JSON "
    "with exactly these keys: \"sentiment\" (string), \"confidence\" (float 0-1), "
    "and \"reason\" (one sentence string). Do not include any text outside the JSON.\n\n"
    f"Review: \"{review}\""
)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": structured_prompt}]
)
raw = response.choices[0].message.content.strip()
print("Raw response:", raw)

try:
    parsed = json.loads(raw)
    print("Sentiment:", parsed["sentiment"])
    print("Confidence:", parsed["confidence"])
    print("Reason:", parsed["reason"])
except json.JSONDecodeError:
    print("Could not parse JSON. Raw response for debugging:")
    print(raw)

# Prompt Q6 - Delimiters
print("\nPrompt Q6 - Delimiters (instructions text):")
user_text = (
    "First boil a pot of water. Once boiling, add a handful of salt and the "
    "pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."
)
prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
print(response.choices[0].message.content)

print("\nPrompt Q6 - Delimiters (non-instruction text):")
non_instruction_text = (
    "The aurora borealis is a natural light display caused by charged solar particles "
    "colliding with gases in Earth's atmosphere."
)
prompt2 = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{non_instruction_text}```
"""
response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt2}]
)
print(response2.choices[0].message.content)

# Delimiters help prevent prompt injection — without them, a malicious or unexpected
# user input could blend into the instruction text and change the model's behavior.
# By wrapping user content in triple backticks (or XML tags, etc.) the model can clearly
# distinguish "instructions" from "data to process."


# --- Local Models with Ollama ---

# Ollama Q1
# Terminal command run:
#   ollama run qwen3:0.6b "Explain what a large language model is in two sentences."
#
# Ollama output (pasted from terminal):
"""
Large language models (LLMs) are AI systems trained on vast amounts of text data to
understand and generate human-like language. They use deep learning techniques,
particularly transformer architectures, to predict and produce coherent text based on
input prompts.
"""

print("\nOllama Q1 - Same prompt via OpenAI API (gpt-4o-mini):")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain what a large language model is in two sentences."}]
)
print(response.choices[0].message.content)

# Differences observed:
# The OpenAI response was more polished and precise in its wording. The Ollama/qwen3:0.6b
# response was coherent but slightly more generic, which is expected from a much smaller model.
#
# Advantage of running locally: complete data privacy — no text leaves your machine,
# and there are no per-token API costs.
#
# Disadvantage of running locally: smaller local models generally produce lower-quality
# outputs than large hosted models, and you need sufficient hardware (RAM/GPU) to run them.
