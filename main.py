import torch
import tiktoken
from GPTModel import GPTModel
from setup import BASE_CONFIG
from utility_functions import Provide_output

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_and_optimizer = torch.load("gpt2-small124M-sft.pth")
model = GPTModel(BASE_CONFIG)

# model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
model.load_state_dict(torch.load("gpt2-small124M-sft.pth"))
model.eval()
instruction = str(input("Please provide instruction for the model: "))

email = {'instruction': instruction, 
         'input': ''}

print(Provide_output(email, model, tokenizer, device,BASE_CONFIG))