import numpy as np 
import torch 
import torch.nn as nn
from src.state.triagestate import TriageState, triage
from src.utility.helperfucn import load_model, triage_rules, generate_reasoning, get_ollama_response

class triageNode:
    def __init__(self,llm):
        self.llm = llm

    
    #-- Here is the classification node----
    def classification_node(self, state:triage) -> TriageState:
        model, scaler, label_encoder = load_model()
        vitals =  state['vitals']
        features = np.array([[vitals['heart_rate'][0], vitals['respiratory_rate'][0], vitals['body_temp'][0],
                          vitals['spo2'][0], vitals['sbp'][0], vitals['dbp'][0], vitals['age'][0],
                          vitals['gender'][0], vitals['weight'][0], vitals['height'][0]]])
        
        scaled = scaler.transform(features)
        tensor_input =  torch.tensor(scaled, dtype=torch.float32)


        with torch.no_grad():
            output = model(tensor_input)
            prob = torch.sigmoid(output).numpy().flatten()[0]
            predicted = 1 if prob >= 0.5 else 0
            label = label_encoder.inverse_transform([predicted])[0]

        state["classification"] = f"{label} ({prob:.2f})"
        return state

    #-----This will provide the current category and the priority for the patients-----
    def node_rule_based_category(self, state:triage)->TriageState:
        #-- Here is the rule based category node----
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

    #--- Here the Llm will generate the reasoing what has happend-----
    def node_generating_reasoning(self, state:triage)->TriageState:
        #-- Here is the reasoning node----
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


    ##---- Here the treatment plan will be generated as the inital vitals and injury condition of the patient
    def node_generate_treatment(self,state: triage) -> TriageState:

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
    
