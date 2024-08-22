from transformers import GPT2TokenizerFast
from src.model import Decoder
from src.config import Config
import torch


def GenerateText(text: str):
    model = Decoder(Config).to(Config.device)
    model.load_state_dict(torch.load(
        './Epoch= 3_minigpt2.pt', map_location=torch.device('cpu')))

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    prompt = tokenizer.encode(
        text, return_tensors='pt').to(Config.device)
    generated_text = model.generate(
        model, prompt, max_tokens=Config.block_size)
    generated_text = tokenizer.decode(generated_text.tolist()[0])
    print(generated_text)
