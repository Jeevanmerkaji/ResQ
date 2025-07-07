import requests
import torch.nn as nn
from huggingface_hub import hf_hub_download
import joblib
import torch


# ----- Rule-based Triage Engine -----
def triage_rules(hr, sbp, dbp, spo2, rr):
    if sbp < 90 and hr > 130:
        return "RED", "Immediate"
    elif spo2 < 92 or rr > 24:
        return "YELLOW", "Delayed"
    elif hr < 40 or spo2 < 80:
        return "BLACK", "Expectant"
    else:
        return "GREEN", "Minor"


#--------This will help me build the reasoning aroung the vitals--------
def generate_reasoning(injury, hr, sbp, spo2, rr, triage):
    ollama_url = "http://localhost:11434/v1/chat/completions"
    prompt = (
        f"A soldier has the following vital signs:\n"
        f"- Heart rate: {hr:.1f} bpm\n"
        f"- Systolic BP: {sbp:.1f} mmHg\n"
        f"- SpO2: {spo2:.1f}%\n"
        f"- Respiratory Rate: {rr:.1f} breaths/min\n"
        f"Injury: {injury}\n"
        f"Triage category: {triage}\n\n"
        f"Explain the reasoning behind this triage decision in a clinical tone."
    )
    messages=[
        {"role": "system", "content": "You are a medical triage assistant."},
        {"role": "user", "content": prompt}
    ]
    model_name = "llama3"
    data = {
        "model": model_name,
        "messages": messages,
    }
    response = requests.post(ollama_url, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code}, {response.text}"


#--------This will help me in building the response-------------
def get_ollama_response(prompt: str, temperature=0.7, max_tokens=300,model_name ="llama3") -> str:
    ollama_url = "http://localhost:11434/v1/chat/completions"
    messages = [
        {"role": "system", "content": "You are a helpful battlefield medic assistant."},
        {"role": "user", "content": prompt}
    ]
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(ollama_url, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code}, {response.text}"

#------------ Model Architecture---------
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

## Load the trained model from the hugging face model hub
def load_model():
    scaler_path = hf_hub_download(repo_id="Jeevan-1998/triage_model", filename="scaler.pkl")
    label_encoder_path = hf_hub_download(repo_id="Jeevan-1998/triage_model", filename="label_encoder.pkl")

    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)


    model_path = hf_hub_download(repo_id="Jeevan-1998/triage_model", filename="human_vital_sign_model.pth")

    # Load the model weights
    model = DNN(input_size=10, hidden_size=128, output_size=1)  # Adjust input_size if needed
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Set the model to evaluation mode
    
    
    return model, scaler, label_encoder