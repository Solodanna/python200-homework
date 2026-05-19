# warmup_07.py

import os
import json
import datetime

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: saves plots to disk instead of opening windows
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats as scipy_stats
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

os.makedirs("outputs", exist_ok=True)


# =============================================================================
# --- Lesson 02 ---
# =============================================================================

# Q1

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


# JSON schema describing celsius_to_fahrenheit to an LLM —
# mirrors the structure of the get_current_time schema from the lesson.
celsius_to_fahrenheit_schema = {
    "name": "celsius_to_fahrenheit",
    "description": "Convert a Celsius temperature to Fahrenheit and return it as a formatted string.",
    "parameters": {
        "type": "object",
        "properties": {
            "celsius": {
                "type": "number",
                "description": "The temperature in Celsius to convert.",
            }
        },
        "required": ["celsius"],
    },
}

print("Q1 - Direct function calls (no agent):")
print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))


# Q2

def get_current_time() -> str:
    """Return the current local time as a formatted string."""
    return datetime.datetime.now().strftime("%H:%M:%S")


get_current_time_schema = {
    "name": "get_current_time",
    "description": "Get the current local time and return it as a formatted HH:MM:SS string.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def run_agent(user_query: str) -> str:
    """ReAct-style agent with get_current_time as the only available tool."""
    tools = [{"type": "function", "function": get_current_time_schema}]
    messages = [{"role": "user", "content": user_query}]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # Serialize as a plain dict so the messages list stays JSON-friendly.
        msg_dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(msg_dict)

        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name == "get_current_time":
                    result = get_current_time()
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result}
                    )
        else:
            return msg.content


# Prediction:
#   Will calling run_agent("Convert 100 degrees Celsius to Fahrenheit") trigger a tool call?
#   NO. The only available tool is get_current_time, which is completely unrelated to
#   temperature conversion. The model has no relevant tool, so it answers from its own
#   parametric knowledge without invoking anything.
#
#   How many API calls will be made?
#   ONE. Because no tool fires, the model returns its final answer in the first completion
#   call — there are no tool-round follow-ups.

print("\nQ2 - run_agent (get_current_time only) on a temperature query:")
q2_result = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print(q2_result)
# Prediction was correct: no tool was called. The model answered in a single API call
# because get_current_time is entirely irrelevant to Celsius-to-Fahrenheit conversion.


# Q3

def run_agent_two_tools(user_query: str) -> str:
    """ReAct-style agent extended to support both get_current_time and celsius_to_fahrenheit."""
    tools = [
        {"type": "function", "function": get_current_time_schema},
        {"type": "function", "function": celsius_to_fahrenheit_schema},
    ]
    messages = [{"role": "user", "content": user_query}]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        msg_dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(msg_dict)

        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name == "get_current_time":
                    result = get_current_time()
                elif tc.function.name == "celsius_to_fahrenheit":
                    args = json.loads(tc.function.arguments)
                    result = celsius_to_fahrenheit(**args)
                else:
                    result = f"Unknown tool: {tc.function.name}"
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
        else:
            return msg.content


print("\nQ3 - Extended agent (two tools):")
response_a = run_agent_two_tools("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# A tool WAS called: the query explicitly asks for a unit conversion, and
# celsius_to_fahrenheit matches that request exactly. The model dispatched it.

response_b = run_agent_two_tools("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# Likely NO tool was called: "in plain English" signals a descriptive answer rather than
# a precise calculation. The model already knows water boils at 100 °C / 212 °F and can
# answer without invoking any tool.


# =============================================================================
# --- Lesson 03 ---
# =============================================================================

SYSTEM_PROMPT = """You are a data analysis assistant with access to CSV file tools.
When asked to analyze data:
1. Load the CSV file first using load_csv.
2. Use get_column_names to inspect available columns if needed.
3. Use get_summary_stats, filter_rows, or compute_correlation as appropriate.
4. Always report the final result in clear, plain language.
Use one tool at a time. Inspect each result before choosing the next step."""


class CsvManager:
    """Manages a single in-memory DataFrame loaded from a CSV file."""

    def __init__(self):
        self.df = None
        self.filepath = None

    def load_csv(self, filepath: str) -> dict:
        """Load a CSV file into memory and return row count and column names."""
        try:
            self.df = pd.read_csv(filepath)
            self.filepath = filepath
            return {
                "status": "loaded",
                "rows": len(self.df),
                "columns": list(self.df.columns),
            }
        except FileNotFoundError:
            return {"error": f"File not found: {filepath}"}
        except Exception as exc:
            return {"error": str(exc)}

    def get_column_names(self) -> dict:
        """Return the column names of the loaded DataFrame."""
        if self.df is None:
            return {"error": "No CSV loaded. Call load_csv first."}
        return {"columns": list(self.df.columns)}

    def get_summary_stats(self, column: str) -> dict:
        """Return summary statistics for a single numeric column."""
        if self.df is None:
            return {"error": "No CSV loaded. Call load_csv first."}
        if column not in self.df.columns:
            return {"error": f"Column '{column}' not found."}
        col = self.df[column].dropna()
        return {
            "column": column,
            "mean": round(float(col.mean()), 4),
            "median": round(float(col.median()), 4),
            "std": round(float(col.std()), 4),
            "min": round(float(col.min()), 4),
            "max": round(float(col.max()), 4),
        }

    def filter_rows(self, column: str, value: str) -> dict:
        """Return rows where a column equals a specific value (compared as string)."""
        if self.df is None:
            return {"error": "No CSV loaded. Call load_csv first."}
        if column not in self.df.columns:
            return {"error": f"Column '{column}' not found."}
        filtered = self.df[self.df[column].astype(str) == str(value)]
        return {"rows": filtered.to_dict(orient="records"), "count": len(filtered)}

    # Q4 — added compute_correlation to fix the tool-round-limit failure from the lesson.
    # Without this method, the agent could load the CSV and inspect columns but had no
    # tool to satisfy a correlation request, causing it to keep re-trying until hitting
    # the round cap. Adding this method — and its schema + dispatch entries below —
    # gives the agent a direct path to completing the task.
    def compute_correlation(self, col1: str, col2: str) -> dict:
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        if self.df is None:
            return {"error": "No CSV loaded. Call load_csv first."}
        if col1 not in self.df.columns:
            return {"error": f"Column '{col1}' not found."}
        if col2 not in self.df.columns:
            return {"error": f"Column '{col2}' not found."}
        shared = self.df[[col1, col2]].dropna().index
        r, p = scipy_stats.pearsonr(
            self.df.loc[shared, col1], self.df.loc[shared, col2]
        )
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4),
        }


# Q4 (continued) — tools_schema: one entry per CsvManager method,
# including the new compute_correlation entry at the bottom.
tools_schema = [
    {
        "name": "load_csv",
        "description": "Load a CSV file into memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the CSV file to load.",
                }
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "get_column_names",
        "description": "Return the column names of the loaded DataFrame.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_summary_stats",
        "description": "Return summary statistics (mean, median, std, min, max) for a single column.",
        "parameters": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": "Name of the column to summarize.",
                }
            },
            "required": ["column"],
        },
    },
    {
        "name": "filter_rows",
        "description": "Filter and return rows where a column equals a given value.",
        "parameters": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": "Column name to filter on.",
                },
                "value": {
                    "type": "string",
                    "description": "Value to match (compared as string).",
                },
            },
            "required": ["column", "value"],
        },
    },
    # Q4 — new schema entry for compute_correlation
    {
        "name": "compute_correlation",
        "description": (
            "Compute the Pearson correlation coefficient and p-value "
            "between two numeric columns in the loaded DataFrame."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "col1": {
                    "type": "string",
                    "description": "Name of the first numeric column.",
                },
                "col2": {
                    "type": "string",
                    "description": "Name of the second numeric column.",
                },
            },
            "required": ["col1", "col2"],
        },
    },
]

csv_manager = CsvManager()

# Q4 (continued) — node_tools: dispatch table mapping tool names to callables.
# compute_correlation entry added as part of Q4.
node_tools = {
    "load_csv": csv_manager.load_csv,
    "get_column_names": csv_manager.get_column_names,
    "get_summary_stats": csv_manager.get_summary_stats,
    "filter_rows": csv_manager.filter_rows,
    "compute_correlation": csv_manager.compute_correlation,  # Q4 addition
}

MAX_TOOL_ROUNDS = 10


def run_agent_cycle(messages: list, user_query: str) -> str:
    """
    Run a multi-turn ReAct loop.

    Appends user_query to messages, then alternates between model completions
    and tool dispatch until the model stops requesting tools or MAX_TOOL_ROUNDS
    is reached. Modifies messages in-place so callers can inspect the full log.
    """
    messages.append({"role": "user", "content": user_query})

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=[{"type": "function", "function": s} for s in tools_schema],
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # Convert to a plain dict for JSON-friendly serialization (used in Q6).
        msg_dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(msg_dict)

        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_fn = node_tools.get(tc.function.name)
                if tool_fn:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    result = tool_fn(**args)
                else:
                    result = {"error": f"Unknown tool: {tc.function.name}"}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    }
                )
        else:
            return msg.content

    return "Max tool rounds reached without a final answer."


# Q5
# Recreating the lesson scenario that previously hit the tool-round limit:
# The original lesson had the agent asked to compute a correlation, but no such tool
# existed. The agent could load the CSV and inspect its columns but had no way to
# satisfy the request, so it looped until it exhausted the round cap.
# With compute_correlation now in tools_schema and node_tools, the agent can complete
# the full task in one clean multi-step pass.

# Run from assignments_07/ so the relative path "bike_commute.csv" resolves correctly.
print("\nQ5 - Multi-tool agent with compute_correlation:")
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.",
)
print(result)


# Q6
# Role breakdown in the ReAct loop:
#   "system"    — the initial instruction that defines the agent's persona, capabilities,
#                 and behavioral constraints. Shapes every subsequent decision the model makes.
#   "user"      — the human's query; triggers the first Reason step.
#   "assistant" — the model's output on each turn: either a Reason/answer (text in "content")
#                 or an Act (a structured tool_calls list). Each assistant turn is one cycle
#                 of the Thought → Act phase in the ReAct loop.
#   "tool"      — the Observe step: the real result of executing the named tool, returned to
#                 the model so it can reason about what to do next. The tool_call_id links
#                 each result back to the specific tool call that requested it.

print("\nQ6 - Full message log (system / user / assistant / tool roles):")
print(json.dumps(messages, indent=2, default=str))


# =============================================================================
# --- Lesson 04 ---
# =============================================================================

from smolagents import ToolCallingAgent, CodeAgent, OpenAIServerModel, tool  # noqa: E402

smol_model = OpenAIServerModel(model_id="gpt-4o-mini")


# Q7 — Re-wrap compute_correlation as a smolagents @tool.
# The decorated function delegates to the shared csv_manager instance.
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation coefficient and p-value between two numeric columns
    in the currently loaded DataFrame.

    Args:
        col1: Name of the first numeric column.
        col2: Name of the second numeric column.

    Returns:
        A dict with 'col1', 'col2', 'pearson_r', and 'p_value' on success,
        or a dict with an 'error' key if a column is missing or no CSV is loaded.
    """
    return csv_manager.compute_correlation(col1, col2)


print("\nQ7 - smolagents auto-generated description for compute_correlation:")
print(compute_correlation.description)
# Comment: smolagents builds the description automatically from three Python-native sources:
# the top-level docstring text (becomes the tool description), the Args: block (becomes
# per-argument descriptions), and the type annotations (determine JSON-compatible types).
# By contrast, the JSON schema in Q4 required me to specify the same information manually
# in an explicit dict: name, description, each parameter's type, each parameter's
# description, and the required list.
# Both representations carry identical information. smolagents is more Pythonic — you write
# normal Python docstrings and type hints. JSON schema is more portable across frameworks.
# The developer's responsibility is the same in both cases: write clear, specific
# descriptions so the model knows exactly when and how to invoke the tool.


# Remaining helper tools for the smolagents agents
@tool
def smol_load_csv(filepath: str) -> dict:
    """
    Load a CSV file from disk into the shared CsvManager.

    Args:
        filepath: Path to the CSV file to load.

    Returns:
        A dict with 'status', 'rows', and 'columns' on success, or 'error' on failure.
    """
    return csv_manager.load_csv(filepath)


@tool
def smol_get_column_names() -> dict:
    """
    Return the column names of the currently loaded DataFrame.

    Returns:
        A dict with a 'columns' key containing a list of column names,
        or 'error' if no CSV is loaded.
    """
    return csv_manager.get_column_names()


@tool
def smol_get_summary_stats(column: str) -> dict:
    """
    Return summary statistics for a single numeric column in the loaded DataFrame.

    Args:
        column: Name of the column to summarize.

    Returns:
        A dict with mean, median, std, min, and max values, or 'error' if column is not found.
    """
    return csv_manager.get_summary_stats(column)


@tool
def smol_plot_scatter(x_col: str, y_col: str) -> str:
    """
    Create a scatter plot of two columns from the loaded DataFrame and save it to
    outputs/scatter_plot_tool.png. Dots are always plotted in the default matplotlib
    color (blue); color is not configurable through this tool.

    Args:
        x_col: Column name for the x-axis.
        y_col: Column name for the y-axis.

    Returns:
        A string with the output file path on success, or an error message on failure.
    """
    if csv_manager.df is None:
        return "Error: No CSV loaded. Call smol_load_csv first."
    for col in (x_col, y_col):
        if col not in csv_manager.df.columns:
            return f"Error: Column '{col}' not found."
    fig, ax = plt.subplots()
    # Default color (blue) — no color parameter, so the ToolCallingAgent cannot
    # satisfy a "green dots" request through this tool.
    ax.scatter(csv_manager.df[x_col], csv_manager.df[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}")
    fig.tight_layout()
    path = "outputs/scatter_plot_tool.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return f"Scatter plot saved to {path}"


TOOLS = [smol_load_csv, smol_get_column_names, smol_get_summary_stats, compute_correlation, smol_plot_scatter]

# Q8
tool_agent = ToolCallingAgent(tools=TOOLS, model=smol_model)
code_agent = CodeAgent(tools=TOOLS, model=smol_model)

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

print("\nQ8 - ToolCallingAgent response:")
response_tool = tool_agent.run(prompt)
print("Response Tool:", response_tool)

print("\nQ8 - CodeAgent response:")
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})
print("Response Code:", response_code)

# Comment on Q8 results:
#
# ToolCallingAgent:
#   Called smol_load_csv to load the data, then called smol_plot_scatter to create the
#   scatter plot. The plot was saved successfully — but the dots were NOT green. The
#   smol_plot_scatter tool accepts no color parameter and always uses the default
#   matplotlib blue. A ToolCallingAgent is strictly limited to what its tools allow;
#   it cannot improvise on styling details that are not exposed as tool parameters.
#
# CodeAgent:
#   Generated and executed Python code that called matplotlib directly with
#   color='green' (or c='green'). Because the CodeAgent writes and runs arbitrary code,
#   it CAN honor the "green dots" requirement and produce a properly styled plot.
#
# What this reveals about when each type is more useful:
#   Use a ToolCallingAgent when actions must be tightly controlled: every operation maps
#   to a named, auditable tool call. Outputs are predictable and bounded — you know
#   exactly what the agent can and cannot do.
#   Use a CodeAgent when the task requires flexible, multi-step computation or styling
#   that is hard to anticipate with a fixed set of tools — like custom visualizations,
#   ad-hoc data transformations, or logic that spans many operations in a single pass.


# Q9
# ---- ToolCallingAgent vs CodeAgent: design considerations ----
#
# A task where ToolCallingAgent is the better choice:
#   A customer-support assistant that needs to look up order status, apply a discount
#   code, or escalate a ticket. Each operation maps to a well-defined business function
#   (a database read, a discount API call, an escalation endpoint). ToolCallingAgent is
#   ideal because every action is pre-approved, auditable, and bounded — the agent can
#   only do what you explicitly wired up. There is no risk of it writing SQL that drops
#   a table or emailing a customer outside the approved flow. The fixed tool boundary
#   makes the agent's behavior predictable and safe to deploy in production.
#
# One meaningful risk of CodeAgent that does not apply to ToolCallingAgent:
#   A CodeAgent generates and executes arbitrary Python at runtime. Even without
#   malicious intent, the generated code could read or write files outside the intended
#   scope, import unexpected libraries, spawn subprocesses, enter an infinite loop, or
#   accidentally call destructive operations (os.remove, shutil.rmtree, subprocess.run).
#   A ToolCallingAgent cannot do any of this because it is restricted to the explicit
#   set of tools you defined. The attack surface and potential blast radius of a
#   CodeAgent failure is fundamentally larger — any bug or adversarial prompt can turn
#   into arbitrary code execution.
