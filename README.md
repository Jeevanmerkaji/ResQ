# Battlefield Triage Intelligence System (BTIS)

AI-Powered Triage & Trauma Care for Frontline Combat Casualties

The Battlefield Triage Intelligence System (BTIS) is an agentic AI workflow designed to automate and accelerate combat casualty assessment. By integrating deep learning, rule-based triage logic, and large language model (LLM) reasoning, BTIS delivers reliable, explainable, and NATO-compliant triage decisions in seconds, even in offline battlefield environments.

---

## Features

- Deep learning for injury severity prediction
- NATO-standard rule-based triage for prioritization
- LLM-powered clinical reasoning for explainable decisions
- Battlefield-adapted treatment planning under resource constraints
- Offline-first design for disconnected environments
- Complete audit trail for operational transparency

---

## Table of Contents

1. [Technical Architecture](#technical-architecture)
2. [Core Components](#core-components)
3. [Data Flow](#data-flow)
4. [Key Innovations](#key-innovations)
5. [Outputs](#outputs)
6. [Installation](#installation)
7. [Usage](#usage)
8. [License](#license)

---

## Technical Architecture

BTIS uses a multi-stage AI pipeline orchestrated with LangGraph to process soldier vitals and injury descriptions into actionable triage categories, clinical reasoning, and treatment protocols.

### High-Level Workflow

1. Input: Soldier ID, injury description, time-series vitals
2. Processing:
   - Injury severity prediction
   - Rule-based triage assignment
   - Explainable reasoning generation
   - Treatment plan generation
3. Output: Structured JSON with triage category, reasoning, and treatment steps

---

## Core Components

| Component            | Technology                               | Function                                                                                     |
|-----------------------|-----------------------------------------|----------------------------------------------------------------------------------------------|
| Classification Model  | PyTorch DNN (Hugging Face Hub)         | Predicts injury severity from 10+ vital signs (HR, SpO₂, BP, etc.).                         |
| Rule Engine           | NATO-compliant triage logic             | Assigns RED / YELLOW / GREEN / BLACK categories based on vitals thresholds.                 |
| Reasoning Generator   | Llama 3 (local Ollama instance)         | Explains triage decisions in clinical terms for medic trust and auditing.                   |
| Treatment Planner     | Llama 3 + RAG (battlefield medicine KB) | Generates step-by-step treatment plans adapted to resource constraints.                     |
| Orchestration         | LangGraph StateGraph                    | Coordinates multi-step workflow with robust error handling.                                 |

---

## Data Flow

### Input

- Soldier ID
- Injury description
- Time-series vitals (HR, SpO₂, BP, etc.)

### Processing Pipeline

1. DNN Classification
   - Outputs injury severity probability
   - Example: "Critical (0.87)"
2. Rule Triage
   - Assigns NATO category
   - Example: "RED / Immediate"
3. LLM Reasoning
   - Clinical justification
   - Example: "Tachycardia + hypotension → hemorrhagic shock risk"
4. Treatment Planning
   - Battlefield-adapted recommendations
   - Example: "Apply tourniquet, request drone evac"

### Output

- Structured JSON with:
  - Triage category
  - Injury severity probability
  - Clinical reasoning
  - Step-by-step treatment plan

---

## Key Innovations

### A. Hybrid AI Decision-Making

- DNN + Rules: Combines machine learning pattern recognition with deterministic NATO-standard triage logic.
- Explainability: LLM-powered clinical reasoning bridges the "black box gap," improving medic trust and auditability.

### B. Battlefield-Optimized

- Offline-First: Fully operational without cloud dependencies.
- Resource-Aware Treatment: Plans tailored to available battlefield supplies.

### C. Observability & Compliance

- Audit Trail: All decisions logged with inputs and rationales (Langfuse integration-ready).
- Bias Mitigation: Trained on diverse combat injury datasets for equitable triage performance.

---

## Outputs

Example output (JSON):

\`\`\`json
{
  "soldier_id": "12345",
  "triage_category": "RED",
  "injury_severity_probability": "Critical (0.87)",
  "clinical_reasoning": "Tachycardia + hypotension → hemorrhagic shock risk",
  "treatment_plan": [
    "Apply tourniquet",
    "Request drone evacuation",
    "Administer IV fluids if available"
  ],
  "timestamp": "2025-06-27T14:23:00Z"
}
\`\`\`

---

## Installation

> Note: This repository requires Python 3.10+.

```bash
git clone https://github.com/Jeevanmerkaji/ResQ.git
cd ResQ
```


Install dependencies:


pip install -r requirements.txt


Configure local Ollama LLM instance:


ollama serve
ollama pull llama3


---

## Usage

Run the workflow with sample data:

\`\`\`bash
python btis_main.py --input data/sample_vitals.json
\`\`\`

Example command with custom parameters:

\`\`\`bash
python btis_main.py --soldier_id 12345 --vitals_file vitals.csv --injury_desc "Blast injury"
\`\`\`

Outputs will be saved as JSON in the outputs/ directory.

---

## License

This project is licensed under the MIT License.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## Contact

For questions or collaboration inquiries, please open an issue or contact jeevanms5355@gmail.com.
