"""Warmup 10 exercises."""


# --- LLMs as Transform ---

# Q1
# Parse the string "Jan 5th, 2024" into ISO format: deterministic code is best because date parsing follows fixed, testable rules.
# Classify "my card was charged twice" into billing/technical/general: an LLM is best because intent classification from natural language benefits from semantic understanding.
# Calculate the average of a list of numbers: deterministic code is best because the math is exact and does not require language reasoning.
# Extract the company name from "Sr. Data Eng @ Acme Corp (contract)": an LLM is often best because freeform job-title text is messy and variable across formats.
# Determine whether a product review is more than 100 words: deterministic code is best because word counting is a simple, precise rule.


# Q2
# Problem: "Summarize this product review in a few sentences" returns unstructured text,
# which is hard to parse consistently and can break downstream storage/analytics when field
# boundaries are ambiguous.
# Better prompt for reliable pipelines:
# system = (
#     "You are a review summarization transformer. Return ONLY valid JSON with this exact "
#     "schema: {\"summary\": string, \"sentiment\": \"positive\"|\"neutral\"|\"negative\", "
#     "\"key_issues\": [string]}. Do not include markdown or extra keys."
# )


# Q3
# Sequential runtime: 50,000 records * 1 second per call = 50,000 seconds,
# which is about 13.9 hours.
# Practical scale strategy: process requests concurrently in controlled batches
# (for example async workers with rate-limit aware retries) to increase throughput
# without changing the model.


# --- Azure OpenAI ---

# Q1
# Two specific reasons organizations choose Azure OpenAI over direct OpenAI API:
# 1. Compliance and governance alignment: Azure OpenAI can be deployed inside an
#    organization's Azure environment with enterprise controls (for example RBAC,
#    Azure Policy, private networking options, and centralized monitoring), which
#    helps satisfy internal security and regulatory requirements.
# 2. Unified Azure operations: teams already using Azure can integrate identity,
#    networking, billing, and resource management in one platform, reducing
#    operational overhead compared with managing a separate external API footprint.

# Q2
# The three Azure-specific client initialization parameters are:
# 1. azure_endpoint: the base URL of your Azure OpenAI resource, such as
#    https://<resource-name>.openai.azure.com/.
# 2. api_version: the Azure OpenAI REST API version string to target (for example
#    2024-02-15-preview), which controls available features and request schema.
# 3. azure_deployment: the name of your model deployment in Azure (configured in
#    Azure AI Foundry/Azure OpenAI Studio), which maps requests to a deployed model.

# Q3
# With AzureOpenAI, the model argument uses your deployment name (an alias like
# "gpt4o-mini-prod"), not the base model label like "gpt-4o-mini".
# You find this value in your Azure OpenAI resource under Deployments (Azure AI
# Foundry / Azure OpenAI Studio), where each deployment has a unique name.
