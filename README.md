# Groundcrew

## About

Groundcrew is an LLM Retrieval Augmented Generation (RAG) solution for a code repository, or "chat with your code". Benefits:
- improved code maintenance
- better knowledge management and documentation
- faster engineering onboarding
- more effient surfacing of code issues

Groundcrew is a companion for code generation solutions like GitHub Copilot. Copilot helps you generate code, while Groundcrew reveals code insights.

## Using Groundcrew

You can run Groundcrew against your code repository as described below. More importantly, you leverage the foundational interfaces and approach to build and maintain your own solution for improving developer efficiency.

## Approach

Groundcrew equips an "agent" (LLM) with access to an extensible set of "tools" that are used to answer user questions.  These two foundational components are described below.

### Tools

 Tools augment the LLM's capabilities, giving the LLM intentional and targeted functionality that minimizes generic responses and hallucinations. They also allow the user to structure the interactions with the LLM in a way that's useful for them.

For example, if you show an LLM a Python script from a codebase and ask how to run it, the LLM will tend to answer generically. Create a virtual environment, install the dependencies then run the file with the Python interpreter, etc.  This is correct, but often unhelpful. Where are the dependencies located? What version of Python should you use? Did the developers even intend for you to run that file as a standalone script? In short, the LLM's canned response based on the source code has ignored other relevant context from the project's documentation and configuration files.

Instead, we created an "Installation and Usage" tool that enables the LLM to search the codebase for documenation and setup files. Those specific pieces of information will in turn produce a much more useful to the user.

Here are some other tools have been developed to give the LLM specific and efficient functionality:

- A codebase question answering system that answers generic questions about the code, e.g., "What does function xyz do?";
- A linting tool that will summarize poorly formatted code;
- A docstring general tool for functions and classes;
- A complexity analysis tool that will point the user to parts of the code that may need to be refactored;
- A search capability to find the use of classes and functions across files.

### The Agent

Tool use is coordinated by an "agent" LLM that can choose which tool to use, call the tool, then respond to the user. An agent can call multiple tools in sequence if needed to answer the user's question. In short, the agent is the "interface" that the user interacts with.

As you interact with the agent, all of the questions, tool calls and answers are kept in a working memory. This means you can ask the agent are series of questions in a conversational manner.

### More resources

[Generative AI, Step-by-step]([url](https://www.youtube.com/playlist?list=PL-pTHQz4RcBbJSkWVqZ2YWUCXrLeFPjV6)) - A YouTube playlist that presents each step in building Groundcrew starting with the idea. Contains demos and technical deep dives with the team.
[LLM RAG fundamentals]([url](https://www.youtube.com/playlist?list=PL-pTHQz4RcBbz78Z5QXsZhe9rHuCs1Jw-))

## Installation

Run the following commands to install `groundcrew` in a dedicated Python (Anaconda) environment.

```shell
git clone https://github.com/prolego-team/groundcrew.git
conda env create -f groundcrew/env.yaml
conda activate groundcrew
pip install -e groundcrew
cd groundcrew
```

Next, make sure you have `neo-sophia` cloned and accessible.  If you cloned both `groundcrew` and `neo-sophia` in your home directory then you are good to go.  Otherwise, open `groundcrew/config.yaml` and put the path to `neo-sophia` in the `repository` field on the first row of the file.

> If you want to ask questions about another codebase, simply change the `repository` entry in `config.yaml` to the path of that repository.

## Use Pre-Generated File Descriptions (Prolego internal use only)

The first time you run `groundcrew` on a codebase it will use an LLM to generate summaries of files in the repo.  This can take quite a bit of time.  To skip this step you can download pre-generated files to jump right into the code Q&A.

1. Download `descriptions.pkl` and `tools.yaml` from [this Google drive location](https://drive.google.com/drive/u/1/folders/16CDEMygEX9u-Kon0h-MFGoe5jTQY_Bd6).  Save these files to the `Downloads` on your Mac.
2. From the `groundcrew` directory, run the following commands.

```shell
mkdir .cache
cp ~/Downloads/descriptions.pkl .cache/
cp ~/Downloads/tools.yaml .cache/
```

## Run Groundcrew

To run `groundcrew` you will need your OpenAI API key saved as an environmental variable.  It is not already set, run `export OPENAI_API_KEY=<your private API key>` to set it.

To start the application run `TOKENIZERS_PARALLELISM=false python -m scripts.run` from the `groundcrew` directory.  After everything loads you will be presented with a user prompt.  Type your questions here and wait for the system to respond.

Example questions:

1. What does the ReACT agent do?
2. What function can load PDF files from a folder?
3. Is there functionality to call the OpenAI API?
