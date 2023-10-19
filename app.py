import numpy as np
import pandas as pd
import random
import solara
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('gpt2', padding_side='left')
model = AutoModelForCausalLM.from_pretrained('gpt2')

text1 = solara.reactive("""One, two, three, four, mango""")
@solara.component
def Page():
  with solara.Column(margin="10"):
    solara.Markdown("#Perplexity")
    solara.Markdown("This is an educational tool. For any given passage of text, this tool augments the original text with highlights and annotations that indicate how 'surprising' each token is to the model, as well as which other tokens the model deemed most likely to occur in its place.")
    css = """
    .mystronggreen{
      background-color:#99ff99;
      color:black!important;
      padding:0px;
    }
    .mygreen{
      background-color:#ccffcc;
      color:black!important;
    }
    .myyellow{
      background-color: #ffff99;
      color:black!important;
    }
    .myorange{
      background-color: #ffcc99;
      color:black!important;
    }
    .myred{
      background-color:#ffcab0;
      color:black!important;
    }
    """
    solara.InputText("Enter text and press enter when you're done:", value=text1, continuous_update=False)
    if text1.value != "":
      with solara.VBox():
        with solara.HBox(align_items="stretch"):
          tokens = tokenizer.encode(text1.value, return_tensors="pt")
          tokens = torch.cat((torch.tensor([tokenizer.eos_token_id]), tokens[0])).reshape(1,-1)
          for i in np.arange(0,len(tokens[0])-1):
            outputs = model.generate(tokens[0][:i+1].reshape(1,-1), max_new_tokens=1, output_scores=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id)
            scores = F.softmax(outputs.scores[0], dim=-1)
            top_10 = torch.topk(scores, 10)
            df = pd.DataFrame()
            a = scores[0][tokens[0][i+1]]
            b = top_10.values
            df["probs"] = list(np.concatenate([a.reshape(-1,1).numpy()[0], b[0].numpy()]))
            diff = 100*(df["probs"].iloc[0]-df["probs"].iloc[1])
            if np.abs(diff)<1:
              color = "mystronggreen"
            elif np.abs(diff)<10:
              color = "mygreen"
            elif np.abs(diff)<20:
              color = "myorange"
            elif np.abs(diff)<30:
              color = "myyellow"
            else:
              color = "myred"
            df["probs"] = [f"{value:.2%}" for value in df["probs"].values]
            aux = [tokenizer.decode(tokens[0][i+1])] + [tokenizer.decode(top_10.indices[0][i]) for i in range(10)]
            df["predicted next token"] = aux
            solara_df = solara.DataFrame(df, items_per_page=11)
            with solara.Tooltip(solara_df, color="white"):
              solara.Style(css)
              solara.Text(f"{tokenizer.decode(tokens[0][i+1])}|", classes=[f"{color}"])
Page()
