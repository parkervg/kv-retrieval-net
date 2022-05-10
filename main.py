import json
import pickle
import uuid
from pathlib import Path

from expiringdict import ExpiringDict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pretty_html_table import build_table
from pydantic import BaseModel
from torch.utils.data import DataLoader

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
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
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

model_dir = Path("./resources/glove/")


def _load_model(dataset):
    base_model = api_serving_utils.load_model(dataset, device="cpu")
    return api_serving_utils.load_state(model_dir / "model.pt", base_model)


kvret_path = "./data/kvret_dataset_public/kvret_{}_public.json"

with open(model_dir / "dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

# dataset = KVRETDataset(
#     train_path=kvret_path.format("train"),
#     dev_path=kvret_path.format("dev"),
#     test_path=kvret_path.format("test"),
#     device=torch.device("cpu"),
#     include_context=True,
#     max_len="longest",
#     reverse_input=True,
#     train_mode = False,
# )
# from src.utils import utils
# def examine_dataset(index):
#     print(utils.ids_to_text(dataset.id2tok, dataset.tok2id["[EOS]"], dataset.test[index].get("input"), reversed=True), "\n",
#           dataset.test[index].get("kb_tuples"))

model = _load_model(dataset)

from src.utils import utils

dataset.train = True
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
evaluate_output = utils.evaluate(
    model,
    dataloader,
    sos_token_id=dataset.tok2id["[SOS]"],
    eos_token_id=dataset.tok2id["[EOS]"],
    id2tok=dataset.id2tok,
)
print(
    " Token-level Accuracy: {:.3f} \t BLEU: {}".format(
        evaluate_output.get("acc"), evaluate_output.get("bleu")
    )
)
dataset.train = False


@app.get("/is_up/", response_class=HTMLResponse)
async def home():
    """
    Used to tell when the API is up on Heroku.
    """
    return JSONResponse({"response": True})


@app.post("/start_session/", response_class=HTMLResponse)
async def start_session(request: SessionRequest):
    scenario_type = request.scenario_type
    request.data_type
    # print(data_type)
    # if data_type == "train":
    #     _dataset = dataset.train
    # elif data_type == "test":
    #     _dataset = dataset.test
    # elif data_type == "dev":
    #     _dataset = dataset.dev
    # else:
    #     raise ValueError(f"Unkown data_type: {data_type}")
    item, kb_df, example_inputs = api_serving_utils.get_kb_state(
        dataset=dataset, scenario_type=scenario_type
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


if __name__ == "__main__":
    item, kb_df = api_serving_utils.get_kb_state(dataset=dataset, scenario_type="poi")
    session_cache["abc"] = {}
    session_cache["abc"]["kb_df"] = kb_df
    session_cache["abc"]["item"] = item
    get_json_prediction()
