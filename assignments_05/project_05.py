# project_05.py
# Mini-Project: Job Application Helper

import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


# --- Task 1: Setup and System Prompt ---

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


# Deliberate choices in the system prompt:
# 1. "career changer" framing — explicitly scoping the user's situation so the model
#    doesn't give generic advice that assumes a traditional career path.
# 2. The reminder to review before submitting is built into the *persona* ("always
#    remind"), not just mentioned once, so the model repeats it naturally throughout
#    the conversation rather than only at the start.
# 3. The instruction to acknowledge industry-norm uncertainty keeps the model from
#    sounding overconfident about norms it may not know.
SYSTEM_PROMPT = """
You are a friendly, practical job application coach specializing in helping career changers
present their transferable skills effectively. The person you are helping is transitioning
from a previous career into a new field and needs help crafting clear, compelling
application materials.

Your responsibilities:
- Help the user rewrite resume bullet points to be specific, results-oriented, and strong.
- Help the user draft cover letter opening paragraphs that are confident and authentic.
- Answer questions about resume structure, application strategy, and professional writing.

Strict behavioral constraints:
- Stay focused exclusively on job application materials. Politely redirect off-topic requests.
- Always remind the user to review and edit your output before submitting it anywhere.
- Acknowledge that you may not know the specific norms of the user's target industry,
  and encourage them to consult industry insiders or job listings to verify your advice.
- Never invent credentials, achievements, or facts that the user has not provided.
- Keep responses concise and actionable.
"""


# --- Task 2: Bullet Point Rewriter ---

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    """Rewrite weak resume bullets into strong, results-oriented versions."""
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
You are a professional resume coach helping a career changer.
Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
Use strong action verbs. Quantify impact where it is clearly implied.
Do not invent facts that are not implied by the original.

Return ONLY a valid JSON list with no other text before or after it.
Each item must have exactly two keys:
  "original" (the original bullet as a string)
  "improved" (your rewritten version as a string)

Bullet points:
```
{bullet_text}
```
"""
    messages = [{"role": "user", "content": prompt}]
    raw = get_completion(messages, temperature=0.3)

    # Strip markdown code fences the model sometimes wraps around JSON
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    result = json.loads(raw)
    return result


# --- Task 3: Cover Letter Generator ---

def generate_cover_letter(job_title: str, background: str) -> str:
    """Generate a confident, specific cover letter opening paragraph."""
    # Few-shot examples chosen to demonstrate two things:
    # 1. How to bridge a non-technical past into a technical role with a narrative arc.
    # 2. How to stay specific and avoid clichés like "I am excited to leverage my skills."
    # The pattern controls tone (confident, first-person, story-driven) and length (3-5 sentences).
    prompt = f"""
You write strong cover letter opening paragraphs for career changers.
The paragraph should be 3-5 sentences: confident, specific, and free of clichés.
Do not use phrases like "I am excited to bring my unique skills" or "I am passionate about."

Here are two examples of the style and tone you should match:

Example 1:
Role: Data Analyst at a healthcare nonprofit
Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
Opening: After seven years as a registered nurse, I've spent my career making decisions
under pressure using incomplete information — which turns out to be excellent training for
data analysis. I recently completed a data analytics program where I built dashboards
tracking patient outcomes across departments. I'm applying to [Company] because your
mission-driven work is exactly where clinical context and technical skill intersect.

Example 2:
Role: Junior Software Engineer at a fintech startup
Background: Ten years in retail banking operations, self-taught Python developer for two years.
Opening: I spent a decade on the operations side of banking, watching technology decisions
get made by people who had never processed a wire transfer or resolved a failed ACH batch.
That frustration turned into curiosity, and two years of self-teaching Python later, I'm
ready to be on the other side of those decisions. I'm applying to [Company] because your
work on payment infrastructure is exactly where my domain expertise and new technical skills
intersect.

Now write an opening paragraph for this person:
Role: {job_title}
Background: {background}
Opening:
"""
    messages = [{"role": "user", "content": prompt}]
    return get_completion(messages, temperature=0.7)


# --- Task 4: Moderation Check ---

def is_safe(text: str) -> bool:
    """Return True if the text passes OpenAI's moderation filter, False if flagged."""
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    if flagged:
        print(
            "\nJob Application Helper: Your message couldn't be processed because it "
            "triggered a content safety filter. Please rephrase your question and try again.\n"
        )
    return not flagged


# --- Task 5: The Chatbot Loop ---

def run_chatbot():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # Skip empty input
        if not user_input:
            continue

        # Moderation check before anything else
        if not is_safe(user_input):
            continue

        # Bullet rewriter branch
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)

            if not raw_bullets:
                print("\nJob Application Helper: No bullets entered. Try again.\n")
                continue

            print("\nJob Application Helper: Rewriting your bullets...\n")
            try:
                rewritten = rewrite_bullets(raw_bullets)
                for item in rewritten:
                    print(f"  Original : {item['original']}")
                    print(f"  Improved : {item['improved']}")
                    print()
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  (Could not parse the response: {e}. Please try again.)\n")

            print("Job Application Helper: Remember to review these before using them!\n")

        # Cover letter branch
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()

            print("\nJob Application Helper: Drafting your opening paragraph...\n")
            opening = generate_cover_letter(job_title, background)
            print(opening)
            print(
                "\nJob Application Helper: Please review and personalize this before "
                "submitting. I may not know all the norms of your target industry.\n"
            )

        # Regular chat turn — append to history to maintain context
        else:
            messages.append({"role": "user", "content": user_input})
            reply = get_completion(messages)
            print(f"\nJob Application Helper: {reply}\n")
            messages.append({"role": "assistant", "content": reply})


# --- Quick smoke tests (run once, then comment out before using the chatbot) ---

def _run_tests():
    print("=== Bullet rewriter test ===")
    bullets = [
        "Helped customers with their problems",
        "Made reports for the management team",
        "Worked with a team to finish the project on time"
    ]
    # These bullets are weak because they use vague verbs ("helped", "made", "worked"),
    # give no indication of scale or impact, and could describe almost any job.
    # The model typically adds specificity (e.g., "Resolved customer inquiries"),
    # stronger verbs, and implied quantification ("ensuring on-time delivery").
    rewritten = rewrite_bullets(bullets)
    for item in rewritten:
        print(f"  Original : {item['original']}")
        print(f"  Improved : {item['improved']}")
        print()

    print("=== Cover letter test ===")
    job_title = "Junior Data Engineer"
    background = (
        "Five years of experience as a middle school math teacher; recently completed "
        "a Python course and built data pipelines using Prefect and Pandas."
    )
    opening = generate_cover_letter(job_title, background)
    print(opening)
    print()

    print("=== Moderation test ===")
    safe_input = "Can you help me improve my resume for a data engineering role?"
    flagged_input = "I want to hurt my manager and burn down the office."
    print(f"Safe input result   : {is_safe(safe_input)}")
    print(f"Flagged input result: {is_safe(flagged_input)}")


# --- Task 6: Ethics Reflection ---
# Format chosen: Option A — Comment block

# Q1 — Bias in the model's advice
# The model was trained on text that skews heavily toward white-collar, English-language,
# and Western professional norms. This means it may subtly favor communication styles
# common in tech, finance, or consulting, and may suggest phrasing that sounds natural
# in those cultures but awkward or overly formal in others. A job-seeker from a
# trade background, a non-English-speaking country, or an industry with different
# conventions (e.g., academia, the arts) might receive advice that feels like it's
# asking them to erase their voice rather than amplify it.

# Q2 — Risks of submitting output without review
# The model can hallucinate plausible-sounding but false details — for example, adding
# a metric ("reduced processing time by 30%") that the user never mentioned. If that
# output is submitted to an employer and the claim surfaces in an interview, it could
# constitute resume fraud, even though the user never intended to deceive anyone.
# Beyond factual errors, the model's tone may not match the user's authentic voice,
# which can create a jarring disconnect when the employer meets the candidate in person.

# Q3 — One guardrail for a professional deployment
# I would add a visible, persistent UI disclaimer — something like a banner that reads
# "Always review AI-generated content before submitting. This tool cannot verify facts
# about your experience." — rather than relying on the model to remind users verbally.
# A text reminder buried in a conversation is easy to ignore; a persistent visual element
# is much harder to miss and sets the right expectations before the user even types anything.


if __name__ == "__main__":
    # Uncomment the line below to run smoke tests, then re-comment before using the chatbot.
    # _run_tests()
    run_chatbot()
