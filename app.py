import os
import datetime
import chainlit as cl
from pydantic import Field

from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    set_default_openai_client,
    function_tool,
)

# Disable tracing for simplicity
set_tracing_disabled(disabled=True)

# 1. Setup OpenAI-compatible client (Gemini here, but works with OpenAI too)
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# 2. Model
llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash", openai_client=external_client
)
set_default_openai_client(external_client)

# ======================================================
# Agents as tools
# ======================================================

search_agent = Agent(
    name="Search Agent",
    instructions="Search the web and return relevant information about the query in bullet points.",
    model=llm_model,
)

summarizer_agent = Agent(
    name="Summarizer Agent",
    instructions="Summarize provided text into concise, clear paragraphs capturing key insights.",
    model=llm_model,
)

synthesizer_agent = Agent(
    name="Synthesizer Agent",
    instructions="Take multiple summaries or research notes and synthesize them into a comprehensive research report with clear sections.",
    model=llm_model,
)

@function_tool
def run_search(query: str) -> str:
    result = Runner.run_sync(search_agent, query)
    return result.final_output

@function_tool
def run_summarizer(text: str) -> str:
    result = Runner.run_sync(summarizer_agent, text)
    return result.final_output

@function_tool
def run_synthesizer(notes: str) -> str:
    result = Runner.run_sync(synthesizer_agent, notes)
    return result.final_output

few_shot_examples = """
Always follow this exact 3-step workflow:
1. Call run_search
2. Call run_summarizer
3. Call run_synthesizer
Never skip steps. Never answer directly without using the tools.
"""

research_manager = Agent(
    name="Research Manager",
    instructions=(
        "You are the Research Manager. Your role is to coordinate the workflow of research. "
        "Follow strictly: (1) run_search, (2) run_summarizer, (3) run_synthesizer.\n\n"
        + few_shot_examples
    ),
    model=llm_model,
    tools=[run_search, run_summarizer, run_synthesizer],
)

# ======================================================
# Beautified Chainlit Front Page
# ======================================================

@cl.on_chat_start
async def start():
    await cl.Message(
        content=(
            "# 📚 Deep Research Assistant\n\n"
            "Welcome to the **AI-powered Research Assistant**! 🚀\n\n"
            "### ✨ Features\n"
            "- Automatically gathers information using `Search Agent`\n"
            "- Summarizes findings with `Summarizer Agent`\n"
            "- Produces a structured research report using `Synthesizer Agent`\n\n"
            "### 📝 How to use\n"
            "Simply type your research topic (e.g., *Impact of AI on Education*), "
            "and I’ll return a polished research report.\n\n"
            "---\n"
            "⚡ *Powered by OpenAI Agents SDK + Chainlit*"
        )
    ).send()

# ======================================================
# Handle Messages
# ======================================================

@cl.on_message
async def main(message: cl.Message):
    result: Runner = Runner.run_sync(research_manager, message.content)
    await cl.Message(content=f"{result.final_output}").send()
