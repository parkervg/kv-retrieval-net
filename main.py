import json
import random
import uuid
from pathlib import Path

import torch
from expiringdict import ExpiringDict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pretty_html_table import build_table
from pydantic import BaseModel

from src.prepare_data import KVRETDataset
from src.utils import api_serving_utils


class ModelRequest(BaseModel):
    text: str
    session_id: str


class SessionRequest(BaseModel):
    scenario_type: str
    data_type: str  # train, test, or dev


class DialogueRequest(BaseModel):
    session_id: str


templates = Jinja2Templates(directory="template")
app = FastAPI()

origins = [
    "https://parkervg.github.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Make sure when deployed, this isn't set to "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Used to cache df knowledge bases and kb_vocab_masks
session_cache = ExpiringDict(max_len=50, max_age_seconds=300, items=None)

model_dir = Path("./resources/bahdanau_base_0.8_teacher_force_0.5_dropout/")


def _load_model(dataset):
    base_model = api_serving_utils.load_model(dataset, device="cpu")
    return api_serving_utils.load_state(model_dir / "model.pt", base_model)


kvret_path = "./data/kvret_dataset_public/kvret_{}_public.json"

dataset = KVRETDataset(
    train_path=kvret_path.format("train"),
    dev_path=kvret_path.format("dev"),
    test_path=kvret_path.format("test"),
    device=torch.device("cpu"),
    include_context=True,
    max_len="longest",
    reverse_input=True,
    train_mode=False,
)

model = _load_model(dataset)


@app.get("/is_up/", response_class=HTMLResponse)
async def home():
    """
    Used to tell when the API is up on Heroku.
    """
    return JSONResponse({"response": True})


@app.post("/start_session/", response_class=HTMLResponse)
async def start_session(request: SessionRequest):
    scenario_type = request.scenario_type
    _dataset = random.choice([dataset.test, dataset.dev])
    item, kb_df, example_inputs = api_serving_utils.get_kb_state(
        dataset=_dataset, scenario_type=scenario_type
    )
    session_id = str(uuid.uuid4())
    session_cache[session_id] = {}
    session_cache[session_id]["kb_df"] = kb_df
    session_cache[session_id]["item"] = item
    session_cache[session_id]["example_inputs"] = example_inputs
    session_cache[session_id]["turn_num"] = 0
    session_cache[session_id]["aggregate_out"] = ""
    print(f"Created new session with id {session_id}")
    print(f"kb: {kb_df}")
    print()
    return JSONResponse(
        {"db_html": build_table(kb_df, color="blue_light"), "session_id": session_id}
    )


@app.post("/get_json_prediction/")
async def get_json_prediction(request: ModelRequest):
    session_id = request.session_id
    item = session_cache[session_id].get("item")
    text = api_serving_utils.canonicalize_input(
        text=request.text, kb_mappings=item.get("kb_mappings")
    )
    if session_id not in session_cache:
        print(f"session_id {session_id} not in session_cache!!")
        return JSONResponse({"response": False})
    print(f"Received request: {text}")
    # Combine with previous inputs/outputs to form chat history
    aggregate_out = session_cache[session_id]["aggregate_out"]
    if aggregate_out:
        text = f"{aggregate_out} * {text}"
    print(f"Submitting query to model with text: \n '{text}'")
    prediction_json, aggregate_out = api_serving_utils.get_prediction_json(
        model=model,
        text=text,
        item=item,
        dataset=dataset,
    )
    session_cache[session_id]["aggregate_out"] = aggregate_out
    print(json.dumps(prediction_json, indent=4))
    return JSONResponse(prediction_json)


@app.post("/get_example_dialogue/")
async def get_example_dialogue(request: DialogueRequest):
    session_id = request.session_id
    item = session_cache[session_id].get("item")
    turn_num = session_cache[session_id]["turn_num"]
    session_cache[session_id]["turn_num"] += 1
    example_dialogue = api_serving_utils.recover_surface_forms(
        session_cache[session_id]["example_inputs"][turn_num], item.get("kb_mappings")
    )
    print(example_dialogue)
    return JSONResponse({"output": example_dialogue})


@app.post("/clear_history/")
async def clear_history(request: DialogueRequest):
    session_id = request.session_id
    print(f"Clearing history for session {session_id}...")
    session_cache[session_id]["aggregate_out"] = ""
    session_cache[session_id]["turn_num"] = 0
    return JSONResponse({"output": True})

@app.post("/delete_cache/")
async def delete_cache(request: DialogueRequest):
    session_id = request.session_id
    print(f"Deleting cache for {session_id}...")
    session_cache.pop(session_id)
