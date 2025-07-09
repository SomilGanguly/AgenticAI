from semantic_kernel.agents import AzureAIAgent
from typing import Dict, List
from dataclasses import dataclass
from datetime import timedelta
from semantic_kernel import Kernel
from src.plugins.file_writer import FileWriter

@dataclass
class AgentConfig:
    name: str
    instructions: str
    model: str
    file_writer: bool = False
    code_interpreter: bool = False
    description: str = ""

class AgentFactory: 
    
    def __init__(self, client, code_interpreter):
        self.client = client
        self.code_interpreter = code_interpreter
        self.agents: Dict[str, AzureAIAgent] = {}
    
    async def create_agent(self, config: AgentConfig) -> AzureAIAgent | None:
        """Create a single agent from configuration"""
        try:
            agent_definition = await self.client.agents.create_agent(
                name=config.name,
                model=config.model,
                instructions=config.instructions,
                description=config.description,
                tools=self.code_interpreter.definitions,
                tool_resources=self.code_interpreter.resources
            )
            agent = AzureAIAgent(client=self.client, definition=agent_definition, kernel=self._file_writer_kernel(config))
            agent.polling_options.run_polling_timeout = timedelta(minutes=30)
            self.agents[config.name] = agent
            return agent
        except Exception as e:
            print(f"Error creating agent {config.name}: {e}")
            return None    
    async def create_agents(self, configs: List[AgentConfig]) -> Dict[str, AzureAIAgent]:
        """Create multiple agents from configurations"""
        agents = {}
        for config in configs:
           temp = await self.create_agent(config)
           if temp is not None:
               agents[config.name] = temp
        return agents
    
    async def cleanup(self) -> None:
        """Clean up all created agents"""
        for agent_name, agent in self.agents.items():
            try:
                await self.client.agents.delete_agent(agent.id)
                print(f"Deleted agent: {agent_name}")
            except Exception as e:
                print(f"Error deleting agent {agent_name}: {e}")
        self.agents.clear()
        
    def _file_writer_kernel(self, config: AgentConfig)->Kernel|None:
        if config.file_writer:
            file_writer_plugin = FileWriter()
            kernel =  Kernel()
            kernel.add_plugin(file_writer_plugin)
            return kernel
        return None