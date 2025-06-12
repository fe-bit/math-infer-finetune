
from typing import List
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage
import re
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START
from sympy import sympify
from sympy.core.sympify import SympifyError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import messages_to_dict
import time

class ReWOO(TypedDict):
    task: str
    plan_string: str
    steps: List
    results: dict
    result: str
    message: List

class ReWOOGeminiModel:
    def __init__(self, model_name="gemini-2.0-flash", temperature: float= 0.0, sleep_time: int=10):
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self.sleep_time = sleep_time

        prompt = """For the following task, make a detailed step-by-step plan to solve the problem.  
For each step, choose **one tool** to retrieve or calculate the necessary information.  
The tool output should be stored in a variable like #E1, #E2, etc., which can be used in later steps.

Tools available:
(1) LLM[input]: A pretrained language model like yourself. Use this to reason through parts of the problem using general knowledge or logic. Input can be a natural language instruction.
(2) Calculator[input]: Use this for basic arithmetic operations (e.g., addition, subtraction, multiplication, division). Input must be a valid math expression like "290 / 2".

**Do not solve the problem directly. Only write the plan and tool inputs.**  
Each step must follow this format:
Plan: [describe the reasoning for the step] #EX = Tool[tool input]

Example:

Task: Marco and his dad went strawberry picking. Marco's dad's strawberries weighed 11 pounds. If together their strawberries weighed 30 pounds, how much did Marco's strawberries weigh?  
Plan: Subtract the dad’s weight from the total to find Marco’s weight. #E1 = Calculator[30 - 11]

Task: Frank was reading through his favorite book. The book had 3 chapters, each with the same number of pages. It has a total of 594 pages. How many pages are in each chapter?  
Plan: Divide the total number of pages by the number of chapters. #E1 = Calculator[594 / 3]

---

Begin!  
Describe your plans with rich details. Each Plan must be followed by exactly one #E.

Task: {task}"""

        prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])
        self.planner = prompt_template | self.model
    
    def __call__(self, task: str, silent:bool=True, wait=True) -> str:
        result = []
        app = self.get_graph()
        for s in app.stream({"task": task}):
            if not silent:
                print(s)
                print("---")
            result.append(s)
            time.sleep(self.sleep_time)
        return result

    def get_plan(self, state: ReWOO):
        task = state["task"]
        result = self.planner.invoke({"task": task})
        regex_pattern = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
        # Find all matches in the sample text
        matches = re.findall(regex_pattern, result.content)
        return {"steps": matches, "plan_string": result.content, "message": messages_to_dict([result])}
    
    def _get_current_task(self, state: ReWOO):
        if "results" not in state or state["results"] is None:
            return 1
        if len(state["results"]) == len(state["steps"]):
            return None
        else:
            return len(state["results"]) + 1

    def tool_execution(self, state: ReWOO):
        """Worker node that executes the tools of a given plan."""
        _step = self._get_current_task(state)
        _, step_name, tool, tool_input = state["steps"][_step - 1]
        _results = (state["results"] or {}) if "results" in state else {}
        
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)
        
        if tool == "Calculator":
            try:
                expr = tool_input
                for k, v in _results.items():
                    # Remove '#' and ensure variables are strings or numbers
                    expr = expr.replace(k, str(v))
                    expr = expr.replace(k.lstrip('#'), str(v))
                # Parse and evaluate safely
                evaluated = sympify(expr).evalf()
                result = str(evaluated)
            except SympifyError as e:
                result = f"SymPy Error: {e}"
            except Exception as e:
                result = f"Error: {e}"
        elif tool == "LLM":
            print(tool_input)
            result = self.model.invoke(tool_input)
            _results[step_name] = str(result)
            return {"result": result.content, "message": messages_to_dict([result])}
        else:
            raise ValueError
        _results[step_name] = str(result)
        return {"results": _results}

    def solve(self, state: ReWOO):
        solve_prompt = """Solve the following task or problem. To solve the problem, we have made a step-by-step Plan and \
retrieved corresponding Evidence for each step. Use them with caution, since long evidence might \
contain irrelevant information.

{plan}

Now solve the question or task according to the provided Evidence above. Respond with the answer \
directly, with no extra words.

Task: {task}
Response:"""
        plan = ""
        for _plan, step_name, tool, tool_input in state["steps"]:
            _results = (state["results"] or {}) if "results" in state else {}
            for k, v in _results.items():
                tool_input = tool_input.replace(k, v)
                step_name = step_name.replace(k, v)
            plan += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]"
        prompt = solve_prompt.format(plan=plan, task=state["task"])
        result = self.model.invoke(prompt)
        
        return {"result": result.content, "message": messages_to_dict([result])}
    
    def _route(self, state):
        _step = self._get_current_task(state)
        if _step is None:
            # We have executed all tasks
            return "solve"
        else:
            # We are still executing tasks, loop back to the "tool" node
            return "tool"
    
    def get_graph(self):
        graph = StateGraph(ReWOO)
        graph.add_node("plan", self.get_plan)
        graph.add_node("tool", self.tool_execution)
        graph.add_node("solve", self.solve)
        
        graph.add_edge("plan", "tool")
        graph.add_edge("solve", END)
        graph.add_conditional_edges("tool", self._route)
        graph.add_edge(START, "plan")
        app = graph.compile()
        return app