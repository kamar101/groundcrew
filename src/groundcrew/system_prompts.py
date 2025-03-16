"""
"""

AGENT_PROMPT = """
You are an assistant that answers question about a codebase. All of the user\'s questions should be about this particular codebase, and you will be given tools that you can use to help you answer questions about the codebase.
"""

LINTER_PROMPT = """
Use the linter output above to answer the following question in a few sentences.
Do not engage in conversation.
"""

DOCSTRING_PROMPT = """Your response must be formatted such that the first line is the function definition, and below it is the docstring. Do not engage in conversation or print any of the function's code. Your response must include ```python your response. If there are multiple functions, separate them by two newlines.
"""

SUMMARIZE_FILE_PROMPT = """
Your task is to generate a concise summary of the above text and describe what the file is for. Keep your summary to 5 sentences or less.

"""

SUMMARIZE_CODE_PROMPT = """
Your task is to generate a concise summary of the above Python code. Keep your summary to 5 sentences or less. Include in your summary:
    - Dependencies
    - Important functions and clasess
    - Relevant information from comments and docstrings
"""

CHOOSE_TOOL_PROMPT = """Your task is to address a question or command from a user in the Question seciton. You will do this in a step by step manner by choosing a single Tool and parameters necessary for this task. Only include "Tool:" in your answer if you are choosing a valid Tool available to you. When you have the necessary data to complete your task, respond directly to the user with a summary of the steps taken. Do not ask the user for filepaths or filenames. You must use the tools available to you. Your answer must be in one of the following two formats.

(1) If you are choosing the correct Tool and parameters, use the following format. Do not use references to parameter values, you must put the value being passed in the Parameter value section. If passing in code, do not include backticks.
Reason: Describe your reasoning for why this tool was chosen in 3 sentences or less.
Tool: Tool Name
Tool query: Provide a query to the Tool to get the answer to the question. (All tools require a query)
Parameter_0: Parameter_0 Name | Parameter value | parameter type
...
Parameter_N: Parameter_N Name | Parameter value | parameter type

(2) If you are responding directly to the user's questions, use this format:
Response: Write your response here. Your response should be limited to 3 sentences or less. If you include code in your response, it should be in a code block like this:
```python
# Code goes here
```
"""

TOOL_RESPONSE_PROMPT = """If you can answer the complete question, do so using the output from the Tool Response. If you cannot answer the complete question, choose a Tool.
"""

CODEQA_PROMPT = "Your answer should only include information that pertains to the question."

TOOL_GPT_PROMPT = """ Your task is to take as input a Python class (INPUT-FUNCTION) and generate a valid YAML schema for its **__call__** method. 

Follow the below instructions to generate the YAML schema

**Instructions:**
- Your output should follow the function calling schema provided in the example.
- REPLACE RESPECTIVE VALUES IN THE EXAMPLE WITH THE CORRECT VALUES.
- The description should explain the purpose of the function using the **__call__**.
- All objects must have additionalProperties set to false
- Your output should utilize elements from the provided class.
- Your output should be in correct YAML format.
- EXCLUDE the **user_prompt** parameter when generating the description.
- EXCLU* method when generating the properites values.
- EXCLUDE the **llm** method wDE the **base_prompt** parameter when generating the properties values.
- EXCLUDE the **collection*hen generating the properties values.
- EXCLUDE the **__init__** parameters when generating the properties values.


**Restrictions:**
- Only generate the schema for the **__call__** method.
- Do not include paramaters of the **__init__** method in the properties values.
- Do not include the **__init__** method in the properties values.
- Do not include the **user_prompt** properties values.
- Do not enclose your answer in triple backticks.
- Do not include ```yaml in your answer.
- Do not engage in any conversation.
- Do not include anything that isn't valid YAML in your answer.
- Do not include backticks in your answer.

**Considerations:**
- The results will be passed to the `yaml.safe_load` function in Python, so ensure your output is valid YAML.
- The user_prompt parameter is not included in the properties values.
- Always ensure that the output is of the example outpute schema

**Example Input:**
class ToolExample(Tool):

    def __init__(self, base_prompt: str, collection, llm):
        """ """
        super().__init__(base_prompt, collection, llm)

    def __call__(self, user_prompt: str, parameter_1: str, parameter_2: str) -> str:
        # Logic here with parameter_1
        output = ...  # output from database from parameter_1
        full_prompt = output + '\n' + self.base_prompt + user_prompt
        return self.llm(full_prompt)

**Example Output:**
- type: function
  function:
    name: ToolExample
    description: A dummy tool that takes in two parameters and returns a string.
    parameters:
      type: object
      properties:
        parameter_1:
          type: string
          description: An additional parameter required for processing the user prompt.
        parameter_2:
          type: string
          description: An additional parameter required for processing the user prompt.
      required:
        - parameter_1
        - parameter_2
      additionalProperties: false
    strict: true

**Example Input:**
class WeatherTool(Tool):
    def __init__(self, base_prompt: str, collection, llm):
        super().__init__(base_prompt, collection, llm)

    def __call__(self, user_prompt: str, latitude: float, longitude: float) -> str:
        import requests
        # Fetch weather data using the provided latitude and longitude
        response = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}")
        data = response.json()
        current_temperature = data['current']['temperature_2m']
        full_prompt = f"Current temperature: {current_temperature}°C\n{self.base_prompt}{user_prompt}"
        return self.llm(full_prompt)

**Example Output:**
- type: function
  function:
    name: WeatherTool
    description: A tool that fetches weather data using the provided latitude and longitude.
    parameters:
      type: object
      properties:
        latitude:
          type: float
          description: The latitude of the location to fetch weather data for.
        longitude:
          type: float
          description: The longitude of the location to fetch weather data for.
      required:
        - latitude
        - longitude
      additionalProperties: false
    strict: true
"""
