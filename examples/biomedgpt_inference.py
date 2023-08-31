import os.path as osp
import sys
path = osp.dirname(osp.abspath(''))
sys.path.append(path)
sys.path.append(osp.join(path, "open_biomed"))
# load biomedgpt model
import json
import torch
from open_biomed.utils import fix_path_in_config
from open_biomed.models.multimodal import BioMedGPTV

config = json.load(open("../configs/encoders/multimodal/biomedgptv.json", "r"))
fix_path_in_config(config, path)
print("Config: ", config)

device = torch.device("mlu:0")
config["network"]["device"] = device
model = BioMedGPTV(config["network"])
ckpt = torch.load("../ckpts/fusion_ckpts/biomedgpt_10b.pth")
model.load_state_dict(ckpt)
model = model.to(device)
model.eval()
print("Finish loading model")
# chat
from open_biomed.utils.chat_utils import Conversation

prompt_sys = "You are working as an excellent assistant in chemistry and molecule discovery. " + \
             "Below a human expert gives the representation of a molecule or a protein. Answer questions about it. "
chat = Conversation(
    model=model, 
    processor_config=config["data"], 
    device=device,
    system=prompt_sys,
    roles=("Human", "Assistant"),
    sep="###",
    max_length=2048
)
chat.append_molecule("CC(=CCC1=CC(=CC(=C1O)CC=C(C)C)/C=C/C(=O)C2=C(C=C(C=C2)O)O)C")
question = "Please describe this molecule."
print("Human: ", question)
chat.ask(question)
print("Assistant: ", chat.answer()[0])
chat.reset()
chat.append_protein("MAKEDTLEFPGVVKELLPNATFRVELDNGHELIAVMAGKMRKNRIRVLAGDKVQVEMTPYDLSKGRINYRFK")
question = "What is the function of this protein?"
print("Human: ", question)
chat.ask(question)
print("Assistant: ", chat.answer()[0])


