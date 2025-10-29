from fastapi import FastAPI
from pydantic import BaseModel


class InputText(BaseModel):
    text: str

app = FastAPI()

@app.get("/")
async def root():
    pass