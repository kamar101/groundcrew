"""
Main agent class interacting with a user
"""
import inspect
import readline

from typing import Any, Callable

from yaspin import yaspin
from yaspin.core import Yaspin
from chromadb import Collection

from groundcrew import agent_utils as autils, system_prompts as sp, utils
from groundcrew.gc_dataclasses import Colors, Config, Tool, ToolMessage
from groundcrew.llm.openaiapi import SystemMessage, UserMessage, AssistantMessage, Message
# from groundcrew.llm.ollama_api import SystemMessage, UserMessage, AssistantMessage, Message


class Agent:
    """
    A class representing an agent that interacts with a user to execute various
    tools based on user prompts.

    Attributes:
        config (dict): Configuration settings for the agent.
        collection (object): The collection or database the agent interacts with.
        chat_llm (object): A chat-based LLM used by the agent for processing
        and interpreting prompts.
        tools (dict): A dictionary of tools available for the agent to use.

    Methods:
        run(): Continuously process user inputs and execute corresponding tools.
        dispatch(user_prompt): Analyze the user prompt and select the
        appropriate tool for response or respond directly if appropriate.
    """

    def __init__(
            self,
            config: Config,
            collection: Collection,
            chat_llm: Callable,
            tools: dict[str, Tool]):
        """
        Constructor
        """
        self.config = config
        self.collection = collection
        self.llm = chat_llm
        self.tools = tools
        self.messages: list[Message] = []
        self.spinner: Yaspin | None = None

        self.colors = {
            'system': Colors.YELLOW,
            'user': Colors.GREEN,
            'agent': Colors.BLUE
        }

    def print(self, text: str, role: str) -> None:
        """
        Helper function to print text with a given color and role.

        Args:
            text (str): The text to print.
            role (str): The role of the text to print.

        Returns:
            None
        """
        print(self.colors[role])
        print(f'[{role}]')
        print(Colors.ENDC)
        print(utils.highlight_code(text, self.config.colorscheme))

    def interact(self, user_prompt: str) -> None:
        """
        Process a user prompt and call dispatch

        Args:
            user_prompt (str): The user's input or question.
        Returns:
            None
        """

        if not self.config.debug:
            self.spinner = yaspin(text='Thinking...', color='green')
            self.spinner.start()

        self.dispatch(user_prompt)

        # Append dispatch messages except for the system prompt
        self.messages.extend(self.dispatch_messages)

        if self.config.debug:
            self.print_message_history(self.messages)
        else:
            self.spinner.stop()

        # Response with the last message from the agent
        self.print(self.messages[-1].content, 'agent')

    def run(self):
        """
        Continuously listen for user input and respond using the chosen tool
        based on the input.
        """
        while True:

            user_prompt = ''

            while user_prompt == '':
                user_prompt = input('[user] > ')
                if '\\code' in user_prompt:
                    print(Colors.YELLOW)
                    print('Code mode activated — type \end to submit')
                    print(Colors.ENDC)
                    user_prompt += '\n'

                    line = input('')

                    while '\\end' not in line:
                        user_prompt += line + '\n'
                        line = input('')

            user_prompt = user_prompt.replace('\\code', '')
            if user_prompt == 'exit' or user_prompt == 'quit' or user_prompt == 'q':
                break

            self.interact(user_prompt)

    def run_tool(self, llm_response: AssistantMessage) -> str:
        """
        Runs the Tool selected by the LLM.

        Args:
            llm_response (Dict): The response from LLM.

        Returns:
            tool_response (str): The response from the tool.
        """

        tool_selection = llm_response.tool_calls[0].function_name
        if tool_selection not in self.tools:
            return 'The LLM tried to call a function that does not exist.'

        tool = self.tools[tool_selection]
        tool_args = self.extract_params(llm_response)

        expected_tool_args = inspect.signature(tool.obj).parameters

        # Filter out incorrect parameters
        new_args = {}
        for param_name, val in tool_args.items():
            if param_name in expected_tool_args:
                new_args[param_name] = val
        tool_args = new_args

        # Add any missing parameters - default to None for now.
        # In the future we'll probably want the LLM to regenerate params
        for param_name in expected_tool_args.keys():
            if param_name == 'user_prompt':
                continue
            if param_name not in tool_args:
                tool_args[param_name] = None

        if self.config.debug:
            print(f'Please standby while I run the tool {tool.name}...')
            print(f'("Tool Args": {tool_args})')
            print()

        tool_response = tool.obj(**tool_args)
        return tool_response

    def interact_functional(self, user_prompt: str) -> str:
        """
        Process a user prompt and call dispatch
        Args:
            user_prompt (str): The user's input or question.
        Returns:
            the system's response
        """

        # spinner = yaspin(text='Thinking...', color='green')
        # spinner.start()

        self.dispatch(user_prompt)
        self.messages.extend(self.dispatch_messages)

        # if self.spinner is not None:
        #     self.spinner.stop()

        content = self.messages[-1].content
        self.print(content, 'agent')
        return content

    def dispatch(self, user_prompt: str) -> None:
        """
        Analyze the user's input and either respond or choose an appropriate
        tool for generating a response. When a tool is called, the output from
        the tool will be returned as the response.

        Args:
            user_prompt (str): The user's input or question.

        Returns:
            None
        """

        if self.spinner is not None:
            self.spinner.stop()

        system_prompt = sp.AGENT_PROMPT

        # the message history involved in solving the current user_prompt
        self.dispatch_messages = []

        user_question = '\n\n### Question ###\n' + user_prompt
        self.dispatch_messages.append(UserMessage(user_question))

        while True:

            self.spinner = yaspin(text='Thinking...', color='green')
            self.spinner.start()
            # Choose tool or get a response
            select_tool_response: Message = self.llm(
                [SystemMessage(system_prompt)] +
                self.messages +
                self.dispatch_messages)
            # Add response to the dispatch messages as an assistant message
            self.dispatch_messages.append(select_tool_response)
            self.spinner.stop()


            if select_tool_response.role != 'assistant': # An error occurred
                break
            elif select_tool_response.role == 'assistant' and select_tool_response.tool_calls is None:
                print('No tool selected')
                break
            else:
                print('Tool selected:',
                      select_tool_response.tool_calls[0].function_name) # Strict Mode (One tool at a time)

            self.spinner = yaspin(
                text='Running ' +
                select_tool_response.tool_calls[0].function_name + '...',
                color='green'
            )
            self.spinner.start()

            # Run Tool
            try:
                tool_response = self.run_tool(select_tool_response)
            except Exception as e:
                tool_response = 'An error occurred while running the tool: ' + \
                    str(e)
                
            self.spinner.stop()
            self.dispatch_messages.append(ToolMessage(
                tool_call_id=select_tool_response.tool_calls[0].tool_call_id, content=tool_response,
                name=select_tool_response.tool_calls[0].function_name))

    def extract_params(
            self,
            llm_response: AssistantMessage) -> dict[str, Any]:
        """
        Extract parameters from LLM response

        Args:
        llm_response (AssistantMessage): The response from LLM.

        Returns:
            args (Dict): A dictionary of arguments to be passed to the function.
        """
        return llm_response.tool_calls[0].function_args

    def run_with_prompts(self, prompts: list[str]):
        """
        Process a list of user prompts and respond using the chosen tool
        based on the input.

        Args:
            prompts (List[str]): List of prompts to be processed by the agent.
        """
        for i, user_prompt in enumerate(prompts):

            if i == 0:
                self.print(user_prompt, 'user')

            self.interact(user_prompt)

            if i < len(prompts) - 1:
                print('Next prompt:')
                self.print(prompts[i + 1], 'user')
                input('\nPress enter to continue...\n')

        self.run()

    def print_message_history(self, messages):
        print('\n', '*' * 50, '\n')
        for message in messages:
            if message.role == 'user':
                color = Colors.GREEN
            elif message.role == 'system':
                color = Colors.RED
            elif message.role == 'assistant':
                color = Colors.BLUE

            print('Role:', message.role)
            print(color)
            print(message.content)
            print(Colors.ENDC)
        print('\n', '*' * 50, '\n')
