from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from groundcrew.api_services.agent_setup import agent_setup, hello_world
from groundcrew.agent import Agent

app = FastAPI()
agent: Agent = agent_setup(
    config='/home/mrrobot/projects/codebase-rag/groundcrew/config.yaml', model='gpt-4-1106-preview')


class Message(BaseModel):
    message: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/message/")
def read_item(message: Message):
    res = agent.interact_functional(message.message)
    return {"message": res}
