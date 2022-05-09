import json
import pickle
import uuid

from expiringdict import ExpiringDict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pretty_html_table import build_table
from pydantic import BaseModel

from src.utils import api_serving_utils


class ModelRequest(BaseModel):
    text: str
    session_id: str
    clear_state: bool


class SessionRequest(BaseModel):
    scenario_type: str


class DialogueRequest(BaseModel):
    session_id: str


templates = Jinja2Templates(directory="template")
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
    "https://parkervg.github.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",  # Make sure when deployed, this isn't set to "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Used to cache df knowledge bases and kb_vocab_masks
session_cache = ExpiringDict(max_len=50, max_age_seconds=300, items=None)


def _load_model(dataset):
    base_model = api_serving_utils.load_model(dataset, device="cpu")
    return api_serving_utils.load_state("./resources/glove/model.pt", base_model)


kvret_path = "./data/kvret_dataset_public/kvret_{}_public.json"

with open("./resources/glove/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)
dataset.train = False

model = _load_model(dataset)


@app.get("/is_up/", response_class=HTMLResponse)
async def home():
    """
    Used to tell when the API is up on Heroku.
    """
    return JSONResponse({"response": True})


@app.post("/start_session/", response_class=HTMLResponse)
async def start_session(request: SessionRequest):
    item, kb_df, example_inputs = api_serving_utils.get_kb_state(
        dataset=dataset, scenario_type=request.scenario_type
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
    clear_state = request.clear_state
    if session_id not in session_cache:
        print(f"session_id {session_id} not in session_cache!!")
        return JSONResponse({"response": False})
    print(f"Received request: {text}")
    if clear_state:
        session_cache[session_id]["aggregate_out"] = ""
    else:
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


if __name__ == "__main__":
    item, kb_df = api_serving_utils.get_kb_state(dataset=dataset, scenario_type="poi")
    session_cache["abc"] = {}
    session_cache["abc"]["kb_df"] = kb_df
    session_cache["abc"]["item"] = item
    get_json_prediction()
