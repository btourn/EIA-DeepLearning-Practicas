

"""
                                          (AWS -> EC2, lambdas, sagemaker)
                                          (Google -> Cloud Computing)
cliente (pagina web, postman (API))  -->  SERVER FLASK  --> predicción
{alcohol, vendor_id, ph, ...}                               {predicción, score}

"""

"""
Comando para crear un env vacio: python3 -m venv .venv
Comando para activar el nuevo virtual env: source .venv/bin/activate
Comando para instalar las dependencias: pip install -r requirements.txt 
Comando para iniciar el server: FLASK_APP=server flask run
"""


import torch
import numpy as np

from flask import Flask, request
app = Flask(__name__)


class NNetWithEmbeddings(torch.nn.Module):

    def __init__(self, number_of_vendors, embedding_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=number_of_vendors, embedding_dim=embedding_dim)
        self.linear_1 = torch.nn.Linear(in_features=(13+embedding_dim), out_features=200, bias=True)
        self.relu_1 = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(in_features=200, out_features=100, bias=True)
        self.relu_2 = torch.nn.ReLU()
        self.linear_3 = torch.nn.Linear(in_features=100, out_features=1, bias=True)

    def forward(self, x, vendor_idx): 
        vendor_emb = self.embedding(vendor_idx) # dim vendor_emb -> (1024, embedding_dim)
        final_input = torch.cat([x, vendor_emb], dim=1) # final_input -> (1024, (13+embedding_dim))
        x = self.linear_1(final_input)
        x = self.relu_1(x)
        x = self.linear_2(x)
        x = self.relu_2(x)
        x = self.linear_3(x)
        return x


model = NNetWithEmbeddings(500, 8)
model.load_state_dict(torch.load("./model_with_emb.torch", map_location=torch.device('cpu')))
model.eval()


@app.route("/")
def index():
    
    x = torch.tensor([0.30578512, 0.16666667, 0.23493976, 0.1809816 , 0.09302326,
       0.09722222, 0.26036866, 0.19838057, 0.34108527, 0.09550562,
       0.20289855, 0.        , 1.        ])
    vendor_idx = torch.tensor([165])

    # CARGAR LOS MIN Y MAX DE LA NORMALIZACION
    # CARGAR EL DICT DE VENDOR ID A VENDOR IDX

    x = x.reshape(1,-1)
    vendor_idx = vendor_idx.reshape(1)

    print(x.shape)
    print(vendor_idx.shape)

    logit = model(x, vendor_idx)
    score = torch.sigmoid(logit).item()

    prediction = True if score > 0.5 else False

    return "La predicción es {} y la probabilidad de que sea True {}".format(prediction, score)
