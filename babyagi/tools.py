import os
from langchain.agents import Tool
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.prompts.prompt import PromptTemplate

todo_prompt = PromptTemplate.from_template(
    "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
)
todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
search = SerpAPIWrapper()
babyagi_tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="TaskList",
        func=todo_chain.run,
        description="useful for when you need to come up with task lists. Input: an objective to create a task list for. Output: a task list for that objective. Please make a list of minimum steps to complete the objective! Please be very clear what the objective is!",
    ),
]