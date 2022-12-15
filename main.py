import json
import os
from typing import Literal, Optional
from uuid import uuid4
from fastapi import FastAPI, HTTPException
import random
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from mangum import Mangum
import predict


app = FastAPI()
handler = Mangum(app)


@app.get("/home")
async def root():
    return {"message": "Welcome to Math Wizard!"}

@app.get('/{full_path}')
def predict_image(full_path: str):
    print('input full path is : ', full_path)
    try:
        parsed_text, eval_text = predict.run_api(full_path)
        displayed_text = f'{parsed_text} = {eval_text}'
        return {'result': displayed_text}
    except Exception as e:
        print('error message is')
        print(e)
        return {'error': 'insert url of image after slash'}


