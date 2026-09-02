from backend.services.llm import get_llm
from backend.observability.telemetry import invoke_structured
from backend.schemas.routing import RoutingDecision


def route_task(user_message: str) -> RoutingDecision:
    llm = get_llm()


    prompt = f"""
You are the PM/Router of an Agentic AI Software Engineer.

Your job is to classify the user's software engineering request.

Available task types:

- CODE_GENERATION
- BUG_FIX
- CODE_REVIEW
- CODE_EXPLANATION
- DOCUMENTATION
- DATA_ANALYSIS
- KNOWLEDGE_SEARCH
- GENERAL

Rules:

1. CODE_GENERATION:
   User wants new code, feature, function, API, or component.

2. BUG_FIX:
   User wants an error, bug, or broken behavior fixed.

3. CODE_REVIEW:
   User wants existing code reviewed or improved.

4. CODE_EXPLANATION:
   User provides or refers to specific existing code and wants
   to understand what that code does.

   Do NOT use CODE_EXPLANATION when the user is asking to locate
   or search for code inside a project. Those requests are
   KNOWLEDGE_SEARCH.

5. DOCUMENTATION:
   User wants README, comments, docs, or technical documentation.

6. DATA_ANALYSIS:
   User wants analysis of CSV, SQL, database, or structured data.

7. KNOWLEDGE_SEARCH:
   User needs information from project files, documentation,
   RAG, or external knowledge sources.

   IMPORTANT:
   If the user asks to FIND, LOCATE, SEARCH, or IDENTIFY WHERE
   something exists in a project/codebase, classify it as
   KNOWLEDGE_SEARCH, not CODE_EXPLANATION.

   Examples:
   - "Find where authentication is implemented in my project"
     → KNOWLEDGE_SEARCH
   - "Which file contains the login logic?"
     → KNOWLEDGE_SEARCH
   - "Search my project for database configuration"
     → KNOWLEDGE_SEARCH.

8. GENERAL:
   Request does not fit the categories above.

Set requires_planning=true when the request requires multiple
engineering steps.

Set requires_knowledge=true when solving the request requires
existing project context, files, documentation, data, or external
information.

Keep reasoning short and specific.

User request:
{user_message}
"""

    return invoke_structured(llm, RoutingDecision, prompt)