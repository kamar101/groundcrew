"""
Wrappers for Ollama API with support for local models.

This code closely mirrors the OpenAI API wrapper code, but uses Ollama.
It defines data structures for messages and tool calls, and provides helper
functions to interact with the Ollama client.
"""

from typing import Callable, Iterable, List, Union
from dataclasses import dataclass
import json

# Assume an 'ollama' package exists that provides a client interface similar to OpenAI.
import ollama


@dataclass(frozen=True)
class ToolCall:
    tool_call_id: str
    tool_type: str
    function_name: str
    function_args: dict


@dataclass(frozen=True)
class SystemMessage:
    content: str
    role: str = 'system'


@dataclass(frozen=True)
class UserMessage:
    content: str
    role: str = 'user'


@dataclass(frozen=True)
class AssistantMessage:
    content: str
    tool_calls: List[ToolCall] | None = None
    role: str = 'assistant'


@dataclass(frozen=True)
class ToolMessage:
    content: Union[str, None]
    tool_call_id: str
    role: str = 'tool'


Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]


def get_ollama_client(config_file: str | None = None) -> ollama.Client:
    """
    Get an Ollama API client for local models.
    If a config_file is provided, configuration will be loaded from that file.
    """
    if config_file:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return ollama.Client(**config)
    else:
        return ollama.Client(host='127.0.0.1:11434')


def toolcall_to_dict(tool_call: ToolCall) -> dict:
    """Convert a ToolCall to a dict that can be embedded in an API message."""
    return {
        'id': tool_call.tool_call_id,
        'type': 'function',
        'function': {
            'name': tool_call.function_name,
            'arguments': json.dumps(tool_call.function_args)
        }
    }


def message_to_dict(message: Message) -> dict:
    """
    Convert a message to a dict that can be passed to the API.
    This is a lightweight alternative to using asdict().
    """
    output_dict = {}
    for key, value in vars(message).items():
        if value is None:
            continue

        if key == 'tool_calls' and value is not None:
            output_dict[key] = [toolcall_to_dict(tc) for tc in value]
        else:
            output_dict[key] = value

    return output_dict


def dict_to_message(message_dict: dict) -> Message:
    """Convert a dict returned by the API to a message object."""
    role = message_dict.get('role')
    if role == 'system':
        return SystemMessage(message_dict['content'])
    elif role == 'user':
        return UserMessage(message_dict['content'])
    elif role == 'assistant':
        tool_calls = None
        if 'tool_calls' in message_dict:
            tool_calls = [
                ToolCall(
                    tc['id'],
                    tc['type'],
                    tc['function']['name'],
                    json.loads(tc['function']['arguments'])
                )
                for tc in message_dict['tool_calls']
            ]
        return AssistantMessage(message_dict['content'], tool_calls)
    elif role == 'tool':
        return ToolMessage(message_dict['content'], message_dict['tool_call_id'])
    else:
        raise ValueError("Unknown message role: " + role)


def message_from_api_response(response: dict) -> AssistantMessage:
    """
    Parse the API response into an AssistantMessage.
    This function expects the response format to be similar to:
    {
      "Message": [
          {"message": {...}}
      ]
    }
    """

    completion = response['message']
    tool_calls = None
    if 'tool_calls' in completion and completion['tool_calls'] is not None:
        tool_calls = [
            ToolCall(
                tc['id'],
                tc['type'],
                tc['function']['name'],
                json.loads(tc['function']['arguments'])
            )
            for tc in completion['tool_calls']
        ]
    return AssistantMessage(completion['content'], tool_calls)


def start_chat(model: str, client: ollama.Client) -> Callable:
    """
    Returns a chat function for interacting with the local model via Ollama.
    This function takes a list of messages and returns the model's response.

    Example usage:

        chat = start_chat('local-model-v1', client)
        messages = [
            SystemMessage("You are a helpful assistant."),
            UserMessage("What is the meaning of 'ollama'?")
        ]
        response = chat(messages)
        print(response)
    """
    def chat_func(messages: List[Message], *args, **kwargs) -> Message:
        if not messages:
            raise ValueError("Messages list cannot be empty")

        start_idx = 1 if isinstance(messages[0], SystemMessage) else 0
        for idx, msg in enumerate(messages[start_idx:]):
            if idx % 2 == 0:
                if not isinstance(msg, (UserMessage, ToolMessage)):
                    raise ValueError("Expected UserMessage or ToolMessage")
            else:
                if not isinstance(msg, AssistantMessage):
                    raise ValueError("Expected AssistantMessage")

        input_messages = [message_to_dict(msg) for msg in messages]
        try:
            response = client.chat(
                messages=input_messages,
                model=model,
                *args,
                **kwargs
            )
            return message_from_api_response(response)
        except ollama.ResponseError as e:
            return UserMessage("There was an API error. Please try again.. " + str(e.error))
    return chat_func


def get_embedding_model(model: str, client: ollama.Client) -> Callable:
    """
    Create an embedding function for the given local model.
    The function takes an iterable of strings and returns a list of embeddings.
    """
    def embedding_func(texts: Iterable[str]) -> List[List[float]]:
        raw_embeddings = client.embeddings.create(
            input=texts,
            model=model
        )['data']
        return [item['embedding'] for item in raw_embeddings]
    return embedding_func


def main():
    # Example usage
    client = get_ollama_client()
    response = client.chat(model='gemma2:2b', messages=[
        {
            'role': 'user',
            'content': 'Why is the sky blue?',
        },
    ])
    print(response)


if __name__ == '__main__':
    main()
