from app.agent.graph import run_agent

print("Script started")
from app.agent.graph import run_agent
print("Agent imported successfully")

questions = [...]

questions = [
    "What optimization method is mentioned in the document for machine learning?",
    "What's the latest news about OpenAI?",
    "What is 45 * 12?",
]

for q in questions:
    answer, tools_used = run_agent(q)
    print(f"\nQ: {q}")
    print(f"Tools used: {tools_used}")
    print(f"A: {answer}")