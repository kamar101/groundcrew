"""
Utility functions
"""
import os
import ast
import inspect
import importlib

from typing import Any, Callable, TypedDict

import yaml
import astunparse

from chromadb import Collection

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import Terminal256Formatter

from groundcrew import constants, system_prompts as sp
from groundcrew.llm import ollama_api, openaiapi
from groundcrew.llm.openaiapi import Message
from groundcrew.gc_dataclasses import Tool


def highlight_code_helper(text: str, colorscheme: str) -> str:
    """
    Highlights code in a given text string.

    Args:
        text (str): The text optionally including code to highlight.
        colorscheme (str): The colorscheme to use for highlighting.
    Returns:
        str: The text with code highlighted.
    """

    start_idx = text.find('```python')
    end_idx = text.find('```', start_idx + 1)

    # No python code found
    if start_idx == end_idx == -1:
        return text

    code = ''
    if '```python' in text:
        code = text.split('```python')[1].split('```')[0]
        code = highlight(
            code,
            PythonLexer(),
            Terminal256Formatter(style=colorscheme, background='dark'))

    return text[:start_idx] + code + text[end_idx + 3:]


def highlight_code(text: str, colorscheme: str) -> str:
    """
    Uses the helper function to highlight code in a given text

    Args:
        text (str): The text optionally including code to highlight.
        colorscheme (str): The colorscheme to use for highlighting.
    Returns:
        str: The text with code highlighted.
    """

    if '```python' not in text:
        return text

    out = highlight_code_helper(text, colorscheme)

    while '```python' in out:
        out = highlight_code_helper(out, colorscheme)

    return out


def build_llm_chat_client(
        model: str = constants.DEFAULT_MODEL, tools: list[dict[str, Any]] = None) -> Callable[[list[Message]], str]:
    """Make an LLM client that accepts a list of messages and returns a response."""
    if 'gpt' in model:
        client = openaiapi.get_openaiai_client()
        chat_session = openaiapi.start_chat(model, client)
    else:
        client = ollama_api.get_ollama_client()
        chat_session = ollama_api.start_chat(model, client)

    def chat(messages: list[Message]) -> Message:
        return chat_session(messages, tools=tools)
    return chat


def build_llm_completion_client(
        model: str = constants.DEFAULT_MODEL) -> Callable[[str], str]:
    """Make an LLM client that accepts a string prompt and returns a response."""
    local_model = False
    if 'gpt' in model:
        client = openaiapi.get_openaiai_client()
        completion = openaiapi.start_chat(model, client)
    else:
        client = ollama_api.get_ollama_client()
        completion = ollama_api.start_chat(model, client)
        local_model = True

    def chat_complete(prompt):
        try:
            if local_model:
                messages = [
                    ollama_api.SystemMessage(
                        "You are a helpful assistant."),
                    ollama_api.UserMessage(prompt)
                ]
                response = completion(messages)
                return response.content
            messages = [
                openaiapi.SystemMessage("You are a helpful assistant."),
                openaiapi.UserMessage(prompt)
            ]
            response = completion(messages)
            return response.content
        except Exception:
            return ''

    return chat_complete


def setup_and_load_yaml(filepath: str, key: str) -> dict[str, dict[str, Any]]:
    """
    Helper function to set up workspace, create a file if it doesn't exist,
    and load data from a YAML file.

    Args:
        filepath (str): The path to the YAML file.
        key (str): The key to extract data from the loaded YAML file.

    Returns:
        dict: A dictionary containing processed data extracted from the YAML
        file. If the file doesn't exist or the data is None, an empty
        dictionary is returned.
    """

    # Create a file if it doesn't exist
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            pass

    # Load data from the file
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    # Process the loaded data
    if data is None:
        return {}

    return {item['function']['name']: item for item in data[key]}


def setup_tools(
        modules_list: list[dict[str, Any]],
        tool_descriptions: dict[str, dict[str, str]],
        collection: Collection,
        llm: Callable,
        working_dir_path: str) -> dict[str, Tool]:
    """
    This function sets up tools by generating a dictionary of Tool objects
    based on the given modules and tool descriptions.

    Args:
        modules_list (list): A list of dictionaries containing modules and the
        functions to be used from those modules.
        tool_descriptions (dict): A dictionary containing descriptions for
        specific tools.

    Returns:
        tools (dict): A dictionary containing Tools with each key being the
        tool name
    """

    # Parameters available to a tool constructor
    params = {
        'collection': collection,
        'llm': llm,
        'working_dir_path': working_dir_path
    }

    tools = {}
    for module_dict in modules_list:
        module_name = module_dict['module']
        module = importlib.import_module(module_name)
        tools_list = module_dict['tools']

        # Loop through tools and generate descriptions
        with open(module.__file__, 'r') as f:
            file_text = ''.join(f.readlines())

        for node in ast.walk(ast.parse(file_text)):
            if isinstance(node, ast.ClassDef) and node.name in tools_list:

                # Actual code of the class
                tool_code = astunparse.unparse(node)

                # Description already generated and in yaml file, so load it.
                if node.name in tool_descriptions:
                    print(f'Loading description for {node.name}...')
                    tool_json_schema = yaml.dump(
                        tool_descriptions[node.name], sort_keys=False)

                else:
                    print(f'Generating description for {node.name}...')

                #     # Generate description of the Tool in YAML format
                #     tool_yaml = convert_tool_str_to_yaml(tool_code, llm)
                #     print("Tool YAML:", tool_yaml)

                # # Convert YAML to a dictionary
                # tool_info_dict = yaml.safe_load(tool_yaml)
                # if isinstance(tool_info_dict, list):
                #     tool_info_dict = tool_info_dict[0]

                # # Remove the user_prompt from the parameters in case the LLM added it
                #
                # if 'user_prompt' in tool_args:
                #     del tool_args['user_prompt']

                tool_constructor = getattr(module, node.name)
                tool_json_schema = tool_constructor.model_json_schema()  # get schema using pydantic
                tool_args = tool_json_schema['properties']

                tool_params = inspect.signature(tool_constructor).parameters

                # Get the expected arguments for the tool constructor
                args = {}
                for param_name, value in params.items():
                    if param_name in tool_params:
                        args[param_name] = value

                # Create an instance of a tool object
                tool_obj = tool_constructor(**args)
                expected_tool_args = inspect.signature(tool_obj).parameters

                # Filter out incorrect parameters for the tool object
                for param_name in tool_args:
                    if param_name not in expected_tool_args:
                        del tool_args[param_name]

                tool_err = f'Tool {tool_obj} must return a string'
                assert inspect.signature(
                    tool_obj).return_annotation is str, tool_err

                # Add the tool to the tools dictionary
                tools[node.name] = Tool(
                    name=node.name,
                    code=tool_code,
                    description=tool_json_schema['description'],
                    params=tool_args,
                    schema=parse_tool_schema(tool_json_schema),
                    obj=tool_obj)

    return tools


def parse_tool_schema(tool_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Parse the schema of a tool into a dictionary format for function calling.

    Args:
        tool (Tool): The tool to parse.
    Returns:
        dict: The schema of the tool in dictionary
    """
    parsed_schema = {}
    description = tool_schema['description']
    title = tool_schema['title']
    properties = tool_schema['properties']

    # Remove default key from schema
    for param, value in properties.items():
        if 'default' in value:
            del value['default']
    # All fields in the scema are required. Optional should be skipped in the Tool class
    required = [property_name for property_name in properties]

    parsed_schema['type'] = "function"  # Constant for now
    parsed_schema['function'] = {
        'name': title,
        'description': description,
        'parameters': {
            'type': 'object',
            'properties': properties,
            'required': required,
            'additionalProperties': False
        },
        'strict': True
    }
    return parsed_schema


def convert_tool_str_to_yaml(function_str: str, llm: Callable) -> str:
    """
    Convert a given tool string to YAML format using a GPT-4 model.

    Args:
        function_str (str): The string representation of a function.

    Returns:
        str: The YAML representation of the given function string.
    """
    return llm(sp.TOOL_GPT_PROMPT + '\n\n----INPUT-FUNCTION---\n\n' + function_str)


def save_tools_to_yaml(tools: dict[str, Tool], filename: str) -> list[dict[str, Any]]:
    """
    Converts a dictionary of tools into YAML format and saves it to a file.

    Args:
        tools (dict): A dictionary containing Tools with each key being the
        tool name
        filename (str): The name of the file to save the YAML data to.

    Returns:
        None
    """

    # Convert the tools dictionary into a list of dictionaries
    tools_list = [tool.schema for tool in tools.values()]

    # Wrap the list in a dictionary with the key 'tools'
    data = {'tools': tools_list}

    # Write the data to the YAML file
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    print(f'Saved {filename}\n')
    return tools_list
