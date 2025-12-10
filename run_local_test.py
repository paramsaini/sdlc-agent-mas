import os
from agent_logic import app # Assuming agent_logic.py is in the same directory
import time

# --- SETUP: Set your Gemini API Key locally ---
# In your terminal, you must run: export GEMINI_API_KEY="[YOUR KEY HERE]"

final_task_goal_mas = """
Goal: Build a scalable and secure User Registration and Profile Viewer application.

COMPONENTS REQUIRED:
1.  'db_schema.sql': SQL table schema defining 'users' (id INTEGER PRIMARY KEY, username TEXT NOT NULL, password_hash TEXT NOT NULL).
2.  'auth_api.py': Python Flask API with two endpoints: '/register' and '/profile'.
3.  'index.html': Frontend page with a registration form and a display area for the user profile.

CRITICAL FAILURE REQUIREMENTS (MUST force correction):
- The API code must initially forget the required HTTP 400 status code handler for invalid inputs. (Forces Security Error/Backend Fix).
- The JavaScript in 'index.html' must initially access the backend but fail to update the DOM content. (Forces Frontend Logic Error/Frontend Fix).
"""

initial_state_mas = {
    "task_goal": final_task_goal_mas,
    "messages": [],
    "iteration": 0,
    "code": "",
    "plan": "",
    "error_log": "",
    "next_component": "schema",
    "backend_code": "", 
    "frontend_code": "", 
    "schema_code": "", 
}

print("\n--- Running FINAL ULTIMATE COMPLEXITY TEST (MAS) ---")
start_time = time.time() 
final_state_mas = app.invoke(initial_state_mas)
end_time = time.time()
total_time = end_time - start_time

print("\n--- ULTIMATE MAS TEST: Final State Summary ---")
print(f"Total Time Taken: {total_time:.2f} seconds")
print(f"Final Iterations: {final_state_mas['iteration']}")
print(f"Execution Status: {'SUCCESS' if 'SUCCESS' in final_state_mas['error_log'] else 'FAILURE'}")
print(f"Execution Result: {final_state_mas['error_log']}")
print(f"Final Code:\n--- DB Schema ---\n{final_state_mas['schema_code']}\n\n--- Backend API ---\n{final_state_mas['backend_code']}\n\n--- Frontend UI ---\n{final_state_mas['frontend_code']}")