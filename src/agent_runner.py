
from src.i18n import i18n
from browser_use.agent.service import Agent
from src.utils.agent_state import AgentState
from src.agent.custom_agent import CustomAgent
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.controller.custom_controller import CustomController
import os

class AgentRunner:
    def __init__(self, config, browser_manager, agent_state):
        self.config = config
        self.browser_manager = browser_manager
        self.agent_state = agent_state

    async def run(self, llm):
        browser, context = await self.browser_manager.get_browser_and_context()

        if self.config.agent_type == "org":
            agent = Agent(
                task=self.config.task,
                llm=llm,
                use_vision=self.config.use_vision,
                browser=browser,
                browser_context=context,
                max_actions_per_step=self.config.max_actions_per_step,
                tool_calling_method=self.config.tool_calling_method,
            )
        elif self.config.agent_type == "custom":
            controller = CustomController()
            agent = CustomAgent(
                task=self.config.task,
                add_infos=self.config.add_infos,
                use_vision=self.config.use_vision,
                llm=llm,
                browser=browser,
                browser_context=context,
                controller=controller,
                system_prompt_class=CustomSystemPrompt,
                agent_prompt_class=CustomAgentMessagePrompt,
                max_actions_per_step=self.config.max_actions_per_step,
                agent_state=self.agent_state,
                tool_calling_method=self.config.tool_calling_method,
            )
        else:
            raise ValueError(f"Invalid agent type: {self.config.agent_type}")

        history = await agent.run(max_steps=self.config.max_steps)

        history_file = os.path.join(self.config.save_agent_history_path, f"{agent.agent_id}.json")
        agent.save_history(history_file)

        return history, history_file
