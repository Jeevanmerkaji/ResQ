from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pymupdf as fitz
import openai
import os
import streamlit as st
from dotenv import load_dotenv
from typing import List, Tuple
import tensorflow as tf
from pydantic import BaseModel
import numpy as np
import torch
import torch.nn as nn
import joblib
import torch.optim as optim
from huggingface_hub import hf_hub_download
import requests


load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# ----- Input Schema -----
class Vitals(BaseModel):
    heart_rate: List[float]
    blood_pressure: List[Tuple[float, float]]  # (systolic, diastolic)
    spo2: List[float]
    resp_rate: List[float]


class Vitals_Classification(BaseModel):
    heart_rate : List[float]
    respiratory_rate : List[float]
    body_temp : List[float]
    spo2 : List[float]
    sbp : List[float]
    dbp :List[float]
    age : List[int]
    gender: List[int]
    weight : List[float]
    height : List[float]

class SoldierRequest(BaseModel):
    soldier_id: str
    vitals: Vitals
    injury_description: str


class SoldierRequest_classify(BaseModel):
    soldier_id: str
    vitals: Vitals_Classification
    # injury_description: str

# ----- LSTM Forecasting Model -----
def build_lstm_model(input_shape=(3, 5)):  # 3 timesteps, 5 features
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(5)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Define the DNN model
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
    
classifymodel = load_model()
# Initialize model with correct input shape
model = build_lstm_model()
dummy_input = np.random.rand(1, 3, 5).astype(np.float32)  # (batch, timesteps, features)
model(dummy_input)  # Initialize weights

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

# ----- LLM-based Reasoning -----
def generate_reasoning(injury, hr, sbp, spo2, rr, triage):
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
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a medical triage assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=150
    )
    return response['choices'][0]['message']['content']

# ----- Complete System Implementation -----
class TriageSystem:
    def __init__(self):
        self.model = build_lstm_model()
        self.classifymodel = load_model()
        # self.llm = ChatOpenAI(temperature=0.7, model="gpt-4")
        self.model_name = "llama3"
        
    def predict_vitals(self, vitals: Vitals) -> Tuple[List[float], str]:
        """Predict next vitals using LSTM model"""
        # Prepare input data (using last 3 readings)
        hr = np.array(vitals.heart_rate[-3:]).reshape(1, -1, 1)
        sbp = np.array([x[0] for x in vitals.blood_pressure[-3:]]).reshape(1, -1, 1)
        dbp = np.array([x[1] for x in vitals.blood_pressure[-3:]]).reshape(1, -1, 1)
        spo2 = np.array(vitals.spo2[-3:]).reshape(1, -1, 1)
        rr = np.array(vitals.resp_rate[-3:]).reshape(1, -1, 1)
        
        # Combine all 5 features
        X = np.concatenate([hr, sbp, dbp, spo2, rr], axis=2)
        
        # Predict next values
        prediction = self.model.predict(X)[0]
        pred_hr, pred_sbp, pred_dbp, pred_spo2, pred_rr = prediction
        
        return prediction, f"Predicted Vitals - HR: {pred_hr:.1f}, SBP: {pred_sbp:.1f}, DBP: {pred_dbp:.1f}, SpO2: {pred_spo2:.1f}, RR: {pred_rr:.1f}"
    

    def predict_triage_category(self, vitals: Vitals_Classification) -> Tuple[str,float,str,List[float]]:
        print("""Predicting the triage category using the classification model""")

        # Extract the latest vitals
        # hr = vitals.heart_rate[-1]
        # rr = vitals.respiratory_rate[-1]
        # bt = vitals.body_temp[-1]
        # spo2 = vitals.spo2[-1]
        # sbp = vitals.sbp[-1]
        # dbp = vitals.dbp[-1]
        # age = vitals.age[-1]
        # gender = vitals.gender[-1]
        # weight = vitals.weight[-1]
        # height = vitals.height[-1]


        hr = vitals.heart_rate
        rr = vitals.respiratory_rate
        bt = vitals.body_temp
        spo2 = vitals.spo2
        sbp = vitals.sbp
        dbp = vitals.dbp
        age = vitals.age
        gender = vitals.gender
        weight = vitals.weight
        height = vitals.height
        

        # Prepare input features
        input_features = np.array([[hr[0], rr[0], bt[0], spo2[0], sbp[0], dbp[0], age[0], gender[0], weight[0], height[0]]])
        # input_features = np.array([[hr, rr, bt, spo2, sbp, dbp, age, gender, weight, height]])
        # Load model, scaler, and label encoder from self.classifymodel
        model, scaler, label_encoder = self.classifymodel


        # Normalize input
        input_scaled = scaler.transform(input_features)
        # Convert to tensor
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            raw_scores = output.squeeze().tolist()
            prob = torch.sigmoid(output).numpy().flatten()
            predicted_class = 1 if prob >= 0.5 else 0
            

        # Decode label (optional if you have label_encoder)
        label = label_encoder.inverse_transform([predicted_class])[0]
        # label = label_encoder.inverse_transform(predicted_class)
        
        print(f"========================label is==============={label}============")
        return label, f"{prob[0]:.2f}", f"Model-based triage category: {label} (confidence: {prob[0]:.2f})", [raw_scores]

    def get_ollama_response(self, prompt: str, temperature=0.7, max_tokens=300,model_name ="llama3") -> str:
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

    def determine_triage(self, vitals: Vitals, injury: str) -> dict:
        """Full triage workflow"""
        # Get current vitals (last reading)
        current_hr = vitals.heart_rate[-1]
        current_sbp, current_dbp = vitals.blood_pressure[-1]
        current_spo2 = vitals.spo2[-1]
        current_rr = vitals.resp_rate[-1]
        
        # Predict future vitals
        # pred_vitals, pred_text = self.predict_vitals(vitals)
        # pred_hr, pred_sbp, pred_dbp, pred_spo2, pred_rr = pred_vitals
        

        #prediction from the classification model
        # label, prob, text, output  =  self.predict_triage_category()
        

        # Apply triage rules
        current_category, current_priority = triage_rules(
            current_hr, current_sbp, current_dbp, current_spo2, current_rr
        )
        
        # pred_category, pred_priority = triage_rules(
        #     pred_hr, pred_sbp, pred_dbp, pred_spo2, pred_rr
        # )
        
        # Generate reasoning
        current_reasoning = generate_reasoning(
            injury, current_hr, current_sbp, current_spo2, current_rr, 
            f"{current_category} ({current_priority})"
        )
        
        # pred_reasoning = generate_reasoning(
        #     injury, pred_hr, pred_sbp, pred_spo2, pred_rr,
        #     f"{pred_category} ({pred_priority})"
        # )
        
        # Generate treatment plan
        treatment_prompt = (
            f"Generate a battlefield treatment plan for:\n"
            f"- Current vitals: HR {current_hr:.1f}, BP {current_sbp:.1f}/{current_dbp:.1f}, "
            f"SpO2 {current_spo2:.1f}%, RR {current_rr:.1f}\n"
            # f"- Predicted vitals: HR {pred_hr:.1f}, BP {pred_sbp:.1f}/{pred_dbp:.1f}, "
            # f"SpO2 {pred_spo2:.1f}%, RR {pred_rr:.1f}\n"
            f"- Injury: {injury}\n"
            f"- Current triage: {current_category} ({current_priority})\n"
            # f"- Predicted triage: {pred_category} ({pred_priority})\n\n"
            "Provide a step-by-step treatment plan considering battlefield constraints."
        )
        
        # treatment_response = self.llm.invoke(treatment_prompt)
        # treatment_plan = treatment_response.content
        treatment_plan = self.get_ollama_response(treatment_prompt, temperature=0.7,max_tokens=300,model_name="llama3")
        return {
            "current_vitals": {
                "values": {
                    "heart_rate": current_hr,
                    "blood_pressure": (current_sbp, current_dbp),
                    "spo2": current_spo2,
                    "resp_rate": current_rr
                },
                "category": current_category,
                "priority": current_priority,
                "reasoning": current_reasoning
            },
            # "predicted_vitals": {
            #     "values": {
            #         "heart_rate": pred_hr,
            #         "blood_pressure": (pred_sbp, pred_dbp),
            #         "spo2": pred_spo2,
            #         "resp_rate": pred_rr
            #     },
            #     "category": pred_category,
            #     "priority": pred_priority,
            #     "reasoning": pred_reasoning
            # },


            "treatment_plan": treatment_plan,
            # "prediction_text": pred_text
        }


