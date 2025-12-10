# --- COMPLETE FINAL MAS AGENT LOGIC ---

from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import re
import os
from langgraph.graph import StateGraph, END

# --- 1. NEW STATE DEFINITION (Multi-File/Component Tracking) ---
class SDLC_AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] 
    task_goal: str  
    plan: str = ""              
    backend_code: str = ""      
    frontend_code: str = ""     
    schema_code: str = ""       
    error_log: str = ""         
    next_component: str = ""    
    iteration: int = 0  

# --- 2. Utilities (Code Parsing - STABLE FIX) ---
def normalize_output_to_string(content) -> str:
    """Recursively extracts content from lists/messages into a single string."""
    if isinstance(content, list):
        return "".join([normalize_output_to_string(item) for item in content])
    if isinstance(content, AIMessage):
        return content.content or ""
    if isinstance(content, str):
        return content
    return str(content)

def extract_code(content) -> str:
    """Safely extracts code from LLM output by normalizing input and using split()."""
    final_content = normalize_output_to_string(content)

    # Bypass RegEx: Use split() based on the code fences (```)
    parts = final_content.split("```")

    if len(parts) >= 3:
        code_block = parts[1].split('\n', 1)
        if len(code_block) > 1:
            return code_block[1].strip()
        else:
            return code_block[0].strip()
            
    return final_content.strip()

# --- 3. Secure Execution Tool (MOCK) ---
@tool
def code_executor(all_code: str) -> str:
    """Executes the combined code structure, checking for structural and logical errors."""
    
    if 'auth_api.py' not in all_code or 'index.html' not in all_code:
        return "FAILURE: Structural Error: Missing critical application components. FIX: backend"
        
    if 'status = 200' in all_code and 'status = 400' not in all_code:
        return "FAILURE: Security Error: The API logic is missing a critical HTTP 400 status code handler. FIX: backend"

    if 'document.getElementById(' in all_code and 'innerHTML' not in all_code and 'innerText' not in all_code:
        return "FAILURE: Frontend Logic Error: The JavaScript logic is accessing an element but not actively updating the content. FIX: frontend"
    
    return "SUCCESS: All structural, logical, and security requirements verified."

sdlc_tools = [code_executor]

# --- 4. Model Initialization ---
# Ensure GEMINI_API_KEY is set in your local environment
if not os.getenv("GEMINI_API_KEY"):
    print("FATAL ERROR: GEMINI_API_KEY not found. Please set it locally.")
    
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
llm_with_tools = llm.bind_tools(sdlc_tools)

# --- 5. Architect Agent ---
def run_architect(state: SDLC_AgentState) -> dict:
    architect_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the Chief Software Architect. Break down the user goal into a complete, structured plan covering all necessary files. Output the plan ONLY."),
        ("human", "Goal: {task_goal}"),
    ])
    architect_chain = (
        RunnablePassthrough.assign(task_goal=lambda x: x["task_goal"])
        | architect_prompt 
        | llm_with_tools.with_config({"tags": ["architect"]})
    )
    result = architect_chain.invoke(state)
    return {
        "plan": result.content,
        "next_component": "schema", 
        "messages": [("ai", "Architecture Plan Generated. Beginning Schema development.")]
    }

# --- 6. Specialized Coder Agents ---

# IMPORTANT: Ensure normalize_output_to_string is available (from Block 1)

def run_schema_coder(state: SDLC_AgentState) -> dict:
    """Generates the SQL/JSON schema code."""
    mapped_input = RunnablePassthrough.assign(
        plan=lambda x: x.get("plan", ""),
        schema_code=lambda x: x.get("schema_code", ""),
        error_log=lambda x: x.get("error_log", ""),
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the specialized Schema Engineer. Generate ONLY the clean SQL or JSON schema code (e.g., CREATE TABLE commands). Output the code block ONLY, starting and ending with triple backticks (```)."),
        ("human", "Plan: {plan}\nExisting Schema Code: {schema_code}\nError Log: {error_log}"),
    ])
    
    chain = mapped_input | prompt | llm_with_tools
    llm_output = chain.invoke(state)
    
    # *** ABSOLUTE FINAL FIX: INLINE FORCED CAST AND EXTRACTION ***
    # This bypasses the utility function completely by performing the list->string->split operation inline.
    
    # 1. Force the output into a reliable string format
    raw_text_output = normalize_output_to_string(llm_output)
    
    # 2. Extract code using the split method (safe now that it is a string)
    parts = raw_text_output.split("```")
    
    if len(parts) >= 3:
        # Code is guaranteed to be the second part, stripped of the language tag
        extracted_code = parts[1].split('\n', 1)[-1].strip()
    else:
        extracted_code = raw_text_output.strip() # Fallback if no ticks were found

    return {
        "schema_code": extracted_code, # Use the directly extracted, guaranteed string
        "iteration": state["iteration"] + 1,
        "error_log": "",
        "next_component": "backend",
        "messages": [("ai", "Schema Code Generated/Fixed.")]
    }

# --- 7. Security/QA Agent ---
def run_qa_agent(state: SDLC_AgentState) -> dict:
    qa_prompt = ChatPromptTemplate.from_messages([("system", "You are the Security/QA Specialist. Analyze the FAILURE provided by the executor. Your ONLY job is to extract the component to be fixed (MUST be 'backend' or 'frontend') and then generate a single, concise fix instruction. Output the instruction ONLY."), ("human", "Plan: {plan}\nBackend Code:\n{backend_code}\nFrontend Code:\n{frontend_code}\nExecution Result: {error_log}"),])
    qa_input = {"plan": state["plan"], "backend_code": state["backend_code"], "frontend_code": state["frontend_code"], "error_log": state["error_log"]}
    qa_chain = RunnablePassthrough.assign(**qa_input) | qa_prompt | llm_with_tools
    fix_instruction = qa_chain.invoke(qa_input).content
    
    if "FIX: backend" in state["error_log"]:
        next_component = "backend"
    elif "FIX: frontend" in state["error_log"]:
        next_component = "frontend"
    else:
        next_component = "backend" 
    return {"error_log": fix_instruction, "next_component": next_component, "messages": [("ai", f"QA Check failed. Routing fix to {next_component}")]}

# --- 8. Executor Wrapper ---
def run_executor_wrapper(state: SDLC_AgentState) -> dict:
    full_code = f"SCHEMA CODE:\n{state['schema_code']}\nBACKEND CODE:\n{state['backend_code']}\nFRONTEND CODE:\n{state['frontend_code']}"
    execution_result = code_executor.invoke({"all_code": full_code}) 
    return {"error_log": execution_result}

# --- 9. The Central Router ---
def route_system(state: SDLC_AgentState) -> str:
    if "SUCCESS" in state["error_log"]:
        return "end"
    if "FAILURE" in state["error_log"]:
        return "qa_agent"
    
    if state["next_component"] == "schema":
        return "schema_coder"
    if state["next_component"] == "backend":
        return "backend_coder"
    if state["next_component"] == "frontend":
        return "frontend_coder"
        
    if state["error_log"] and state["next_component"] in ["backend", "frontend", "schema"]:
        if state["next_component"] == "schema":
            return "schema_coder"
        if state["next_component"] == "backend":
            return "backend_coder"
        if state["next_component"] == "frontend":
            return "frontend_coder"
    return "executor" 

# --- 10. LangGraph Assembly and Compilation ---
workflow = StateGraph(SDLC_AgentState) 
workflow.add_node("architect", run_architect)
workflow.add_node("schema_coder", run_schema_coder) 
workflow.add_node("backend_coder", run_backend_coder)
workflow.add_node("frontend_coder", run_frontend_coder)
workflow.add_node("executor", run_executor_wrapper)
workflow.add_node("qa_agent", run_qa_agent)

workflow.set_entry_point("architect")
workflow.add_edge("architect", "schema_coder") 
workflow.add_edge("schema_coder", "backend_coder") 
workflow.add_edge("backend_coder", "frontend_coder") 
workflow.add_edge("frontend_coder", "executor") 

workflow.add_conditional_edges("executor", route_system, {"qa_agent": "qa_agent", "end": END})
workflow.add_conditional_edges("qa_agent", route_system, {"schema_coder": "schema_coder", "backend_coder": "backend_coder", "frontend_coder": "frontend_coder",})

app = workflow.compile()
