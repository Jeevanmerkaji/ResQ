import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import json

# Import your existing modules
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, Tuple, List
from pydantic import BaseModel
import torch
import numpy as np
import requests
from huggingface_hub import hf_hub_download
import joblib
import torch.nn as nn
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Try to import transformers (optional for HF fallback)
try:
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ Transformers library not installed. Run 'pip install transformers' for Hugging Face fallback support.")

# Your existing code (DNN class, functions, etc.)
class TriageState(TypedDict):
    soldier_id: str
    vitals: dict
    injury: str
    classification: str
    rule_category: str
    rule_priority: str
    reasoning: str
    treatment: str

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
    try:
        scaler_path = hf_hub_download(repo_id="Jeevan-1998/triage_model", filename="scaler.pkl")
        label_encoder_path = hf_hub_download(repo_id="Jeevan-1998/triage_model", filename="label_encoder.pkl")
        model_path = hf_hub_download(repo_id="Jeevan-1998/triage_model", filename="human_vital_sign_model.pth")

        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(label_encoder_path)

        model = DNN(input_size=10, hidden_size=128, output_size=1)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        
        return model, scaler, label_encoder
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def node_classification_model(state: TriageState) -> TriageState:
    try:
        model, scaler, label_encoder = load_model()
        if model is None:
            state["classification"] = "Model unavailable"
            return state
            
        vitals = state['vitals']
        features = np.array([[vitals['heart_rate'][0], vitals['respiratory_rate'][0], vitals['body_temp'][0],
                              vitals['spo2'][0], vitals['sbp'][0], vitals['dbp'][0], vitals['age'][0],
                              vitals['gender'][0], vitals['weight'][0], vitals['height'][0]]])
        scaled = scaler.transform(features)
        tensor_input = torch.tensor(scaled, dtype=torch.float32)

        with torch.no_grad():
            output = model(tensor_input)
            prob = torch.sigmoid(output).numpy().flatten()[0]
            predicted = 1 if prob >= 0.5 else 0
            label = label_encoder.inverse_transform([predicted])[0]

        state["classification"] = f"{label} ({prob:.2f})"
    except Exception as e:
        state["classification"] = f"Error: {str(e)}"
    return state

def triage_rules(hr, sbp, dbp, spo2, rr):
    if sbp < 90 and hr > 130:
        return "RED", "Immediate"
    elif spo2 < 92 or rr > 24:
        return "YELLOW", "Delayed"
    elif hr < 40 or spo2 < 80:
        return "BLACK", "Expectant"
    else:
        return "GREEN", "Minor"

def node_rule_based_category(state: TriageState) -> TriageState:
    vitals = state['vitals']
    hr = vitals['heart_rate'][-1]
    sbp = vitals['sbp'][-1]
    dbp = vitals['dbp'][-1]
    spo2 = vitals['spo2'][-1]
    rr = vitals['respiratory_rate'][-1]

    cat, pri = triage_rules(hr, sbp, dbp, spo2, rr)
    state["rule_category"] = cat
    state["rule_priority"] = pri
    return state

def get_ollama_response(prompt: str, temperature=0.7, max_tokens=150, model_name="llama3") -> str:
    """Get response from Ollama with Hugging Face fallback"""
    
    # First try Ollama
    try:
        ollama_url = "http://localhost:11434/v1/chat/completions"
        messages = [
            {"role": "system", "content": "You are a helpful battlefield medic assistant. Be concise."},
            {"role": "user", "content": prompt}
        ]
        data = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = requests.post(ollama_url, json=data, timeout=15)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"Ollama unavailable (Status: {response.status_code}), trying Hugging Face...")
            return get_huggingface_response(prompt, max_tokens)
            
    except requests.exceptions.Timeout:
        print("Ollama timeout, trying Hugging Face...")
        return get_huggingface_response(prompt, max_tokens)
    except requests.exceptions.RequestException:
        print("Ollama connection error, trying Hugging Face...")
        return get_huggingface_response(prompt, max_tokens)
    except Exception as e:
        print(f"Ollama error: {e}, trying Hugging Face...")
        return get_huggingface_response(prompt, max_tokens)

def get_huggingface_response(prompt: str, max_tokens: int = 150) -> str:
    """Get response from Hugging Face API as fallback"""
    try:
        import os
        
        
        # Try to use Hugging Face Inference API first (faster)
        hf_token = os.getenv('HUGGINGFACE_API_TOKEN')
        if hf_token:
            try:
                headers = {"Authorization": f"Bearer {hf_token}"}
                api_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
                
                response = requests.post(
                    api_url,
                    headers=headers,
                    json={"inputs": prompt, "parameters": {"max_length": max_tokens}},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get('generated_text', '').replace(prompt, '').strip()
            except Exception as e:
                print(f"HF API error: {e}, trying local model...")
        
        # Fallback to local lightweight model
        try:
            # Use a small, fast model for medical text generation
            model_name = "mistralai/Magistral-Small-2506"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Generate response
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=min(max_tokens, 100),  # Keep it short for speed
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Create a medical context prompt
            medical_prompt = f"Medical Assessment: {prompt}\nResponse:"
            result = generator(medical_prompt, max_length=len(medical_prompt.split()) + max_tokens)
            
            if result and len(result) > 0:
                response_text = result[0]['generated_text']
                # Extract only the response part
                if "Response:" in response_text:
                    return response_text.split("Response:")[-1].strip()
                else:
                    return response_text.replace(medical_prompt, '').strip()
                    
        except Exception as e:
            print(f"Local HF model error: {e}")
            
    except ImportError:
        print("Transformers library not available, using rule-based fallback...")
    except Exception as e:
        print(f"Hugging Face error: {e}")
    
    # Final fallback to rule-based response
    return get_fallback_response(prompt)

def get_fallback_response(prompt: str) -> str:
    """Provide rule-based fallback responses when AI is unavailable"""
    
    if "reasoning" in prompt.lower():
        if "RED" in prompt and "Immediate" in prompt:
            return """🔴 CRITICAL CONDITION: Patient shows signs of shock with low blood pressure (<90 mmHg) and elevated heart rate (>130 bpm). 
            The combination of these vital signs indicates potential hemorrhagic shock or cardiac compromise requiring immediate intervention. 
            Immediate priority is warranted due to life-threatening instability."""
        elif "YELLOW" in prompt and "Delayed" in prompt:
            return """🟡 URGENT CONDITION: Patient has compromised respiratory status (SpO2 <92% or RR >24) indicating potential respiratory distress. 
            While not immediately life-threatening, this requires prompt medical attention to prevent deterioration. 
            Delayed priority allows for stabilization of more critical patients first."""
        elif "GREEN" in prompt and "Minor" in prompt:
            return """🟢 STABLE CONDITION: Patient's vital signs are within acceptable ranges for the current situation. 
            No immediate life-threatening concerns identified. Can safely wait for treatment after higher priority patients."""
        elif "BLACK" in prompt and "Expectant" in prompt:
            return """⚫ EXPECTANT: Patient shows signs incompatible with survival given current resources (HR <40 or SpO2 <80). 
            In battlefield triage, resources must be allocated to those with better survival chances."""
    
    elif "treatment" in prompt.lower():
        if "RED" in prompt:
            return """🏥 IMMEDIATE TREATMENT PROTOCOL:
            1. Secure airway and breathing
            2. Control hemorrhage with pressure/tourniquet
            3. Establish IV access for fluid resuscitation
            4. Monitor vital signs continuously
            5. Prepare for rapid evacuation
            6. Consider blood products if available"""
        elif "YELLOW" in prompt:
            return """🏥 DELAYED TREATMENT PROTOCOL:
            1. Assess and support respiratory function
            2. Monitor vital signs every 15 minutes
            3. Provide oxygen therapy if available
            4. Treat pain and prevent shock
            5. Prepare for evacuation when resources allow"""
        elif "GREEN" in prompt:
            return """🏥 MINOR TREATMENT PROTOCOL:
            1. Basic wound care and dressing
            2. Pain management as needed
            3. Monitor for changes in condition
            4. Document injuries for follow-up care
            5. Self-care instructions if appropriate"""
    
    return "⚠️ AI analysis temporarily unavailable. Using rule-based assessment. Recommend medical officer review."

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
    return get_ollama_response(prompt)

def node_generate_reasoning(state: TriageState) -> TriageState:
    vitals = state['vitals']
    hr = vitals['heart_rate'][-1]
    sbp = vitals['sbp'][-1]
    spo2 = vitals['spo2'][-1]
    rr = vitals['respiratory_rate'][-1]
    injury = state["injury"]
    triage = f"{state['rule_category']} ({state['rule_priority']})"

    reason = generate_reasoning(injury, hr, sbp, spo2, rr, triage)
    state["reasoning"] = reason
    return state

def node_generate_treatment(state: TriageState) -> TriageState:
    vitals = state['vitals']
    injury = state["injury"]
    prompt = (
        f"Generate a battlefield treatment plan for:\n"
        f"Vitals: HR {vitals['heart_rate'][-1]}, BP {vitals['sbp'][-1]}/{vitals['dbp'][-1]}, "
        f"SpO2 {vitals['spo2'][-1]}%, RR {vitals['respiratory_rate'][-1]}\n"
        f"Injury: {injury}\n"
        f"Triage: {state['rule_category']} ({state['rule_priority']})\n"
        f"Provide a step-by-step plan considering battlefield constraints."
    )
    response = get_ollama_response(prompt)
    state["treatment"] = response
    return state

# Build the graph
graph_builder = StateGraph(TriageState)
graph_builder.add_node("classification_patient", node_classification_model)
graph_builder.add_node("rule_logic", node_rule_based_category)
graph_builder.add_node("generate_reasoning", node_generate_reasoning)
graph_builder.add_node("generate_treatment", node_generate_treatment)

graph_builder.set_entry_point("classification_patient")
graph_builder.add_edge("classification_patient", "rule_logic")
graph_builder.add_edge("rule_logic", "generate_reasoning")
graph_builder.add_edge("generate_reasoning", "generate_treatment")
graph_builder.add_edge("generate_treatment", END)

triage_graph = graph_builder.compile()

# Initialize Dash app
app = dash.Dash(__name__)

# Define color scheme for triage categories
triage_colors = {
    'RED': '#dc3545',
    'YELLOW': '#ffc107',
    'GREEN': '#28a745',
    'BLACK': '#343a40'
}

app.layout = html.Div([
    html.Div([
        html.H1("🏥 Medical Triage Dashboard", 
                style={'text-align': 'center', 'color': '#2c3e50', 'margin-bottom': '30px'}),
        
        # Input Form Section
        html.Div([
            html.H3("Patient Information", style={'color': '#34495e', 'margin-bottom': '20px'}),
            
            html.Div([
                html.Div([
                    html.Label("Soldier ID:", style={'font-weight': 'bold'}),
                    dcc.Input(id='soldier-id', type='text', value='S01', 
                             style={'width': '100%', 'padding': '8px', 'margin': '5px 0'})
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Injury Description:", style={'font-weight': 'bold'}),
                    dcc.Textarea(id='injury', value='Gunshot wound to the chest',
                               style={'width': '100%', 'height': '60px', 'padding': '8px', 'margin': '5px 0'})
                ], style={'width': '48%', 'float': 'right'})
            ]),
            
            html.Br(),
            html.H4("Vital Signs", style={'color': '#34495e', 'margin-top': '20px'}),
            
            # Vital signs inputs
            html.Div([
                html.Div([
                    html.Label("Heart Rate (bpm):", style={'font-weight': 'bold'}),
                    dcc.Input(id='hr', type='number', value=130, 
                             style={'width': '100%', 'padding': '8px', 'margin': '5px 0'})
                ], style={'width': '23%', 'display': 'inline-block', 'margin-right': '2%'}),
                
                html.Div([
                    html.Label("Respiratory Rate:", style={'font-weight': 'bold'}),
                    dcc.Input(id='rr', type='number', value=28,
                             style={'width': '100%', 'padding': '8px', 'margin': '5px 0'})
                ], style={'width': '23%', 'display': 'inline-block', 'margin-right': '2%'}),
                
                html.Div([
                    html.Label("Body Temp (°C):", style={'font-weight': 'bold'}),
                    dcc.Input(id='temp', type='number', value=37.0, step=0.1,
                             style={'width': '100%', 'padding': '8px', 'margin': '5px 0'})
                ], style={'width': '23%', 'display': 'inline-block', 'margin-right': '2%'}),
                
                html.Div([
                    html.Label("SpO2 (%):", style={'font-weight': 'bold'}),
                    dcc.Input(id='spo2', type='number', value=88.0, step=0.1,
                             style={'width': '100%', 'padding': '8px', 'margin': '5px 0'})
                ], style={'width': '23%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                html.Div([
                    html.Label("Systolic BP:", style={'font-weight': 'bold'}),
                    dcc.Input(id='sbp', type='number', value=90.0, step=0.1,
                             style={'width': '100%', 'padding': '8px', 'margin': '5px 0'})
                ], style={'width': '23%', 'display': 'inline-block', 'margin-right': '2%'}),
                
                html.Div([
                    html.Label("Diastolic BP:", style={'font-weight': 'bold'}),
                    dcc.Input(id='dbp', type='number', value=60.0, step=0.1,
                             style={'width': '100%', 'padding': '8px', 'margin': '5px 0'})
                ], style={'width': '23%', 'display': 'inline-block', 'margin-right': '2%'}),
                
                html.Div([
                    html.Label("Age:", style={'font-weight': 'bold'}),
                    dcc.Input(id='age', type='number', value=30,
                             style={'width': '100%', 'padding': '8px', 'margin': '5px 0'})
                ], style={'width': '23%', 'display': 'inline-block', 'margin-right': '2%'}),
                
                html.Div([
                    html.Label("Gender:", style={'font-weight': 'bold'}),
                    dcc.Dropdown(id='gender', options=[
                        {'label': 'Male', 'value': 1},
                        {'label': 'Female', 'value': 0}
                    ], value=1, style={'margin': '5px 0'})
                ], style={'width': '23%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                html.Div([
                    html.Label("Weight (kg):", style={'font-weight': 'bold'}),
                    dcc.Input(id='weight', type='number', value=70.0, step=0.1,
                             style={'width': '100%', 'padding': '8px', 'margin': '5px 0'})
                ], style={'width': '23%', 'display': 'inline-block', 'margin-right': '2%'}),
                
                html.Div([
                    html.Label("Height (cm):", style={'font-weight': 'bold'}),
                    dcc.Input(id='height', type='number', value=175.0, step=0.1,
                             style={'width': '100%', 'padding': '8px', 'margin': '5px 0'})
                ], style={'width': '23%', 'display': 'inline-block'})
            ]),
            
            html.Br(),
            html.Button('Run Triage Analysis', id='analyze-btn', n_clicks=0,
                       style={'background-color': '#3498db', 'color': 'white', 'padding': '12px 24px',
                              'border': 'none', 'border-radius': '5px', 'font-size': '16px',
                              'cursor': 'pointer', 'margin': '20px 0'})
        ], style={'background-color': '#ecf0f1', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '20px'}),
        
        # Loading indicator
        dcc.Loading(
            id="loading",
            children=[html.Div(id="loading-output")],
            type="default",
        ),
        
        # Results Section
        html.Div(id='results-section', children=[])
        
    ], style={'max-width': '1200px', 'margin': '0 auto', 'padding': '20px'})
])

@app.callback(
    Output('results-section', 'children'),
    Output('loading-output', 'children'),
    Input('analyze-btn', 'n_clicks'),
    State('soldier-id', 'value'),
    State('injury', 'value'),
    State('hr', 'value'),
    State('rr', 'value'),
    State('temp', 'value'),
    State('spo2', 'value'),
    State('sbp', 'value'),
    State('dbp', 'value'),
    State('age', 'value'),
    State('gender', 'value'),
    State('weight', 'value'),
    State('height', 'value')
)
def update_results(n_clicks, soldier_id, injury, hr, rr, temp, spo2, sbp, dbp, age, gender, weight, height):
    if n_clicks == 0:
        return [], ""
    
    # Create state for analysis
    state = {
        "soldier_id": soldier_id or "Unknown",
        "injury": injury or "Not specified",
        "vitals": {
            "heart_rate": [hr, hr, hr],
            "respiratory_rate": [rr, rr, rr],
            "body_temp": [temp],
            "spo2": [spo2, spo2, spo2],
            "sbp": [sbp, sbp, sbp],
            "dbp": [dbp, dbp, dbp],
            "age": [age],
            "gender": [gender],
            "weight": [weight],
            "height": [height]
        },
        "classification": "",
        "rule_category": "",
        "rule_priority": "",
        "reasoning": "",
        "treatment": ""
    }
    
    try:
        # Run the triage analysis
        result = triage_graph.invoke(state)
        
        # Create visualization
        vital_signs_fig = go.Figure()
        vital_signs_fig.add_trace(go.Scatter(
            x=['Current'], y=[hr], mode='markers+text', name='Heart Rate',
            text=[f'{hr} bpm'], textposition='top center', marker=dict(size=15)
        ))
        vital_signs_fig.add_trace(go.Scatter(
            x=['Current'], y=[sbp], mode='markers+text', name='Systolic BP',
            text=[f'{sbp} mmHg'], textposition='top center', marker=dict(size=15)
        ))
        vital_signs_fig.add_trace(go.Scatter(
            x=['Current'], y=[spo2], mode='markers+text', name='SpO2',
            text=[f'{spo2}%'], textposition='top center', marker=dict(size=15)
        ))
        vital_signs_fig.add_trace(go.Scatter(
            x=['Current'], y=[rr], mode='markers+text', name='Respiratory Rate',
            text=[f'{rr} /min'], textposition='top center', marker=dict(size=15)
        ))
        
        vital_signs_fig.update_layout(
            title='Current Vital Signs',
            xaxis_title='Time',
            yaxis_title='Values',
            height=400,
            showlegend=True
        )
        
        # Triage priority indicator
        triage_color = triage_colors.get(result['rule_category'], '#6c757d')
        
        results_layout = [
            html.H2("🎯 Triage Results", style={'color': '#2c3e50', 'margin-bottom': '20px'}),
            
            # Summary Cards
            html.Div([
                html.Div([
                    html.H4("Triage Category", style={'margin': '0', 'color': '#fff'}),
                    html.H2(f"{result['rule_category']}", style={'margin': '10px 0', 'color': '#fff'}),
                    html.P(f"Priority: {result['rule_priority']}", style={'margin': '0', 'color': '#fff'})
                ], style={'background-color': triage_color, 'padding': '20px', 'border-radius': '10px',
                         'text-align': 'center', 'width': '30%', 'display': 'inline-block', 'margin-right': '3%'}),
                
                html.Div([
                    html.H4("AI Classification", style={'margin': '0', 'color': '#fff'}),
                    html.H3(f"{result['classification']}", style={'margin': '10px 0', 'color': '#fff'})
                ], style={'background-color': '#8e44ad', 'padding': '20px', 'border-radius': '10px',
                         'text-align': 'center', 'width': '30%', 'display': 'inline-block', 'margin-right': '3%'}),
                
                html.Div([
                    html.H4("Patient Info", style={'margin': '0', 'color': '#fff'}),
                    html.P(f"ID: {result['soldier_id']}", style={'margin': '5px 0', 'color': '#fff'}),
                    html.P(f"Age: {age}, Gender: {'Male' if gender == 1 else 'Female'}", 
                          style={'margin': '5px 0', 'color': '#fff'})
                ], style={'background-color': '#34495e', 'padding': '20px', 'border-radius': '10px',
                         'text-align': 'center', 'width': '30%', 'display': 'inline-block'})
            ], style={'margin-bottom': '30px'}),
            
            # Vital Signs Chart
            html.Div([
                dcc.Graph(figure=vital_signs_fig)
            ], style={'background-color': '#fff', 'padding': '20px', 'border-radius': '10px', 
                     'box-shadow': '0 2px 4px rgba(0,0,0,0.1)', 'margin-bottom': '20px'}),
            
            # Detailed Information
            html.Div([
                html.Div([
                    html.H4("🔍 Clinical Reasoning", style={'color': '#2c3e50'}),
                    html.P(result['reasoning'], style={'line-height': '1.6', 'text-align': 'justify'})
                ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),
                
                html.Div([
                    html.H4("🏥 Treatment Plan", style={'color': '#2c3e50'}),
                    html.P(result['treatment'], style={'line-height': '1.6', 'text-align': 'justify'})
                ], style={'width': '48%', 'float': 'right', 'vertical-align': 'top'})
            ], style={'background-color': '#fff', 'padding': '20px', 'border-radius': '10px',
                     'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Injury Information
            html.Div([
                html.H4("🩹 Injury Details", style={'color': '#2c3e50'}),
                html.P(f"Reported Injury: {result['injury']}", style={'font-size': '16px', 'margin': '10px 0'})
            ], style={'background-color': '#fff', 'padding': '20px', 'border-radius': '10px',
                     'box-shadow': '0 2px 4px rgba(0,0,0,0.1)', 'margin-top': '20px'})
        ]
        
        return results_layout, ""
        
    except Exception as e:
        error_layout = [
            html.Div([
                html.H3("⚠️ Analysis Error", style={'color': '#e74c3c'}),
                html.P(f"An error occurred during analysis: {str(e)}", 
                      style={'color': '#e74c3c', 'background-color': '#fdf2f2', 
                             'padding': '15px', 'border-radius': '5px'})
            ])
        ]
        return error_layout, ""

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8050)