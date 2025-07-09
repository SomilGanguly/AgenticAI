from typing import Dict
from semantic_kernel.agents import AzureAIAgent
from semantic_kernel.agents.strategies import SequentialSelectionStrategy, TerminationStrategy
from src.util.AgentFactory import AgentConfig

QUERY_ORCHESTRATOR_NAME="QUERY_ORCHESTRATOR"
QUERY_ORCHESTRATOR_DESCRIPTION="This Agent is responsible for orchestrating queries to other agents. It can handle complex queries by breaking them down into simpler tasks and delegating them to specialized agents. It ensures that the overall query is answered efficiently and accurately."

def __generate_query_orchestrator_instructions(agents: Dict[str, AzureAIAgent]) -> str:
    agent_list = []
    for agent_name, agent in agents.items():
        description = getattr(agent.description, 'description', 'No description available')
        agent_list.append(f"- {agent_name} : {description}")
    
    agents_text = "\n    ".join(agent_list)
    
    return f"""
    You are a query orchestrator agent. Your role is to analyze the conversation history and determine which specialized agent should handle the next part of the task based on the current context and progress.

    INSTRUCTIONS:
    1. Examine the ENTIRE conversation history, including the most recent responses from agents
    2. Identify what has been accomplished so far and what still needs to be done
    3. Look at the LAST agent's output to understand what was just completed
    4. Determine if the task requires another agent or if it's complete
    5. CRITICAL: RESPOND WITH ONLY THE AGENT NAME (no explanation, no additional text)

    DECISION PRINCIPLES:
    - ALWAYS look at what the previous agent just accomplished
    - If an agent just completed their task successfully, consider what comes next in the workflow
    - Don't select the same agent consecutively unless they explicitly indicate more work is needed
    - Match the next required action to the agent whose description best fits that need
    - Consider dependencies between agents (some agents may build upon others' work)
    - If you see that all requirements have been satisfied, respond with "PERFECTUS" to end the conversation
    - If an agent indicates they cannot complete a task or need input from another agent, select the appropriate next agent

    WORKFLOW ANALYSIS:
    - Look at the conversation flow and understand what stage we're at
    - If content was just created, it might need review
    - If content was just reviewed, it might need revision or the task might be complete
    - Consider the logical sequence of the task

    Available agents:
    {agents_text}

    Remember: Your decision should be based on the CURRENT state of the conversation and what the last agent just did. Don't repeat the same agent unless there's a clear reason.
    """
    
def generate_query_orchestrator_config(agents: Dict[str, AzureAIAgent], model: str) -> AgentConfig:
    instructions = __generate_query_orchestrator_instructions(agents)
    
    return AgentConfig(
        name=QUERY_ORCHESTRATOR_NAME,
        description=QUERY_ORCHESTRATOR_DESCRIPTION,
        instructions=instructions,
        model=model,
    )

class ChatSelectionStrategy(SequentialSelectionStrategy):
    """A strategy for determining which agent should take the next turn in the group chat."""
    async def select_agent(self, agents, history):
        agent_name = QUERY_ORCHESTRATOR_NAME
        if(history[-1].name==QUERY_ORCHESTRATOR_NAME):
            agent_name=history[-1].content.strip()
            print(f"Agent selected by orchestrator: {agent_name}")
            return next((agent for agent in agents if agent.name == agent_name), None)
        return next((agent for agent in agents if agent.name == agent_name), None) 

class ApprovalTerminationStrategy(TerminationStrategy):
    """A strategy for determining when the group chat should end."""
    
    async def should_agent_terminate(self, agent, history):
        """Check if the agent should terminate based on the history."""
        return 'PERFECTUS' in history[-1].content.lower() if history else False
