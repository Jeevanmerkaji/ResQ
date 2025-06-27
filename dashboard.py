from dash import Dash, html, dcc, Input, Output,ctx,State,MATCH,ALL
import dash_bootstrap_components as dbc
from dash import dcc
import plotly.express as px
from fpdf import FPDF
import plotly.graph_objects as go
from opcua import Client
import pathlib
import pandas as pd
import datetime
import atexit
import random
import socket
from contextlib import closing
import time
import threading
from OPCUAServer_new import OPCUAClient
from triagelogic import Vitals, TriageSystem, SoldierRequest, Vitals_Classification,SoldierRequest_classify
import fpdf
import base64
import io
import webbrowser
import logging
import os
import dash_html_components as html
import dash
from dash.exceptions import PreventUpdate
import json
# os.makedirs("logs", exist_ok=True)
log_path = os.path.join(os.path.dirname(__file__), 'logs', 'triage_classification.log')
logging.basicConfig(
    filename=log_path,
    filemode="a",  # Append mode
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO  # Or DEBUG for more detailed logs
)

logging.info("Here the Script is starting===================================")


triage_system = TriageSystem()


triage_results = {} 
triage_classification_results ={}

# Initialize app with military dark theme
app = Dash(__name__,
           external_stylesheets=[
               dbc.themes.DARKLY,
               {
                   'href': 'https://fonts.googleapis.com/css2?family=Orbitron&display=swap',
                   'rel': 'stylesheet'
               }
           ],
           meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}],
           suppress_callback_exceptions=True)
app.title = "TACTICAL SOLDIER STATUS DASHBOARD"
server = app.server

history={}
class_name ={}

# Create a global OPCUAClient object for the connection
opc_thread = OPCUAClient()
opc_thread.daemon = True
opc_thread.start()

# Sample Data for Hospital Dashboard
current_admitted = 56
icu_patients = 8
occupancy_rate = 72
admissions_last_24h = [5, 10, 15, 20, 10]  # Mock data points for 24h

patients_info = [
    {"id": "Patient_1", "age": 63, "priority": "High"},
    {"id": "Patient_2", "age": 47, "priority": "Medium"},
    {"id": "Patient_3", "age": 54, "priority": "Low"},
    {"id": "Patient_4", "age": 38, "priority": "Low"},
]

admissions_last_24h = [3, 5, 10, 7, 12]
time_labels = ['12 AM', '6 AM', '12 PM', '3 PM', '6 PM']

# Build frames for animation (one point added at a time)
frames = [
    go.Frame(
        data=[go.Scatter(
            x=time_labels[:k+1],
            y=admissions_last_24h[:k+1],
            mode='lines+markers',
            line=dict(color='#1f77b4', width=4),
            marker=dict(color='#FF6347', size=10, symbol='circle', line=dict(color='black', width=2))
        )],
        name=str(k)
    )
    for k in range(len(admissions_last_24h))
]

admissions_graph = dcc.Graph(
    figure=go.Figure(
        data=[go.Scatter(
            x=[],
            y=[],
            mode='lines+markers',
            line=dict(color='#1f77b4', width=4),
            marker=dict(color='#FF6347', size=10, symbol='circle', line=dict(color='black', width=2))
        )],
        layout=go.Layout(
            title=dict(
                text='',
                x=0.5,
                font=dict(size=30, family='Helvetica, Arial, sans-serif', color='#FF6347'),
                pad=dict(b=20)
            ),
            xaxis=dict(
                title='Time',
                tickvals=time_labels,
                ticktext=time_labels,
                tickangle=45,
                gridcolor='#eee'
            ),
            yaxis=dict(
                title='Admissions',
                gridcolor='#eee'
            ),
            plot_bgcolor='#F6EDF3',
            paper_bgcolor='#ffffff',
            showlegend=False,
            hovermode='closest',
            margin=dict(t=40, b=40, l=40, r=40),
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[dict(
                    label='Play',
                    method='animate',
                    args=[None, {
                        'frame': {'duration': 500, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 300, 'easing': 'linear'}
                    }]
                )]
            )]
        ),
        frames=frames
    )
)

# Sample data for blood pressure
systolic = [120, 125, 130, 135, 140]
diastolic = [80, 85, 90, 92, 95]
time_data = ['12 AM', '6 AM', '12 PM', '3 PM', '6 PM']
patient_info = dbc.Card([
    dbc.CardBody([
        html.H4("Patient Information", style={"color": "#333", "font-weight": "bold"}),
        html.P(f"Name: Marvin McKinney", style={"color": "#666"}),
        html.P(f"Age: 69", style={"color": "#666"}),
        html.P(f"Conditions: Hypertension (HBP HTN), Chronic Obstructive Pulmonary Disease", style={"color": "#666"}),
        html.P(f"Devices: Blood Pressure Monitor, Weighing Scale, Blood Glucose Meter", style={"color": "#666"}),
        html.P(f"Care Team: Alex Martin (CM), Josh Droxi (SCM), Max Dev (Physician)", style={"color": "#666"}),
    ])
], color="info", style={"width": "100%", "borderRadius": "10px"})

def render_soldier_table(soldiers):
    return html.Div([
        html.Div(
            soldier['name'],
            id={'type': 'soldier-entry', 'index': soldier['id']},
            n_clicks=0,
            className="soldier-row"
        )
        for soldier in soldiers
    ])



# Blood Pressure Graph with Thresholds
# bp_graph = dcc.Graph(
#     figure={
#         'data': [
#             go.Scatter(
#                 x=time,
#                 y=systolic,
#                 mode='lines+markers',
#                 name='Systolic',
#                 line=dict(color='orange', width=3),
#                 marker=dict(symbol='circle', size=8, color='orange')
#             ),
#             go.Scatter(
#                 x=time,
#                 y=diastolic,
#                 mode='lines+markers',
#                 name='Diastolic',
#                 line=dict(color='green', width=3),
#                 marker=dict(symbol='circle', size=8, color='green')
#             ),
#             # Thresholds
#             go.Scatter(
#                 x=[time_data[0], time_data[-1]],
#                 y=[130, 130],
#                 mode='lines',
#                 name='Systolic Threshold',
#                 line=dict(color='red', dash='dash')
#             ),
#             go.Scatter(
#                 x=[time_data[0], time_data[-1]],
#                 y=[90, 90],
#                 mode='lines',
#                 name='Diastolic Threshold',
#                 line=dict(color='blue', dash='dash')
#             ),
#         ],
#         'layout': {
#             'title': 'Blood Pressure Readings',
#             'xaxis': {'title': 'Time', 'tickangle': 45},
#             'yaxis': {'title': 'Pressure (mmHg)'},
#             'plot_bgcolor': '#f5f5f5',
#             'paper_bgcolor': '#ffffff',
#             'showlegend': True
#         }
#     }
# )
# Military-style color scheme
COLORS = {
    'background': '#0a0a0a',
    'text': '#00FF41',  # Matrix green
    'panel': '#121212',
    'critical': '#FF0000',
    'warning': '#FFA500',
    'normal': '#00AA00'
}

# Dashboard Layout
app.layout = dbc.Container(fluid=True, children=[
    dcc.Store(id='selected-soldier-data'),
    
    # Modal for soldier details
    dbc.Modal(
        id='soldier-detail-modal',
        is_open=False,
        size="lg",
        children=[
            dbc.ModalHeader(dbc.ModalTitle("SOLDIER DETAILS")),
            dbc.ModalBody(id='modal-body')
        ],
        backdrop='static',
        scrollable=True,
        centered=True
    ),
    
    # Header with military insignia
    dbc.Row([
        dbc.Col(html.Img(src=app.get_asset_url("ResQ.png"), style={"max-height": "60px", "width": "auto"}), width=2),
        dbc.Col(html.H1("TACTICAL SOLDIER MONITORING",
                        style={'color': COLORS['text'], 'font-family': 'Orbitron, sans-serif'}), width=8),
        dbc.Col(html.Div(id='live-clock',
                         style={'color': COLORS['text'], 'font-size': '24px'}),
                width=2)
    ], className="mb-4", style={'border-bottom': f"2px solid {COLORS['text']}"}),


    #for logo----- This is working fine
    # dbc.Row([
    #     dbc.Col(html.Img(src=app.get_asset_url("military_logo.png"), height="50px"), width=2),
    #     dbc.Col(html.H1("Patient Monitoring Dashboard", style={'color': '#333', 'font-family': 'Arial, sans-serif'}), width=8),
    #     dbc.Col(patient_info),
    #     dbc.Col(bp_graph),
    # ], className="mb-4", style={'border-bottom': '2px solid #333'}),
        # Hospital Dashboard Section (Admissions, Priority, etc.)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div("🏥", style={
                            "backgroundColor": "#007bff",
                            "color": "white",
                            "borderRadius": "50%",
                            "width": "50px",
                            "height": "50px",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "fontSize": "24px",
                            "marginBottom": "10px"
                        }),
                        html.H4(f"{current_admitted}", style={"color": "#FFFFFF", "margin": 0}),
                        html.P("Currently Admitted", style={"color": "#BBBBBB", "margin": 0})
                    ], style={"textAlign": "center"})
                ])
            ], style={"backgroundColor": "#1e1e2f", "border": "none", "borderRadius": "12px", "padding": "10px"}),
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div("🛌", style={
                            "backgroundColor": "#dc3545",
                            "color": "white",
                            "borderRadius": "50%",
                            "width": "50px",
                            "height": "50px",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "fontSize": "24px",
                            "marginBottom": "11px"
                        }),
                        html.H4(f"{icu_patients}", style={"color": "#FFFFFF", "margin": 0}),
                        html.P("ICU Patients", style={"color": "#BBBBBB", "margin": 0})
                    ], style={"textAlign": "center"})
                ])
            ], style={"backgroundColor": "#1e1e2f", "border": "none", "borderRadius": "12px", "padding": "10px"}),
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div("📊", style={
                            "backgroundColor": "#28a745",
                            "color": "white",
                            "borderRadius": "50%",
                            "width": "50px",
                            "height": "50px",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "fontSize": "24px",
                            "marginBottom": "10px"
                        }),
                        html.H4(f"{occupancy_rate}%", style={"color": "#FFFFFF", "margin": 0}),
                        html.P("Occupancy Rate", style={"color": "#BBBBBB", "margin": 0})
                    ], style={"textAlign": "center"})
                ])
            ], style={"backgroundColor": "#1e1e2f", "border": "none", "borderRadius": "12px", "padding": "10px"}),
        ], width=3),
    ], justify="center", className="hospital-stats-row"),

    
    # Main content area
    dbc.Row([
        # Map Panel (Hospital Monitoring)
        dbc.Col([
            # html.Div("HOSPITALIZED PATIENT MONITORING", className="panel-header"),
            dbc.Card([
                dbc.CardHeader("HOSPITALIZED PATIENT MONITORING", style= {"textAlign": "center"},className = "panel-header"),
                dbc.CardBody(id="hospital-patient-monitoring-body", children=[
                    html.Div("Loading patient data...", id="hospital-health-status", style={"backgroundColor": "#474A49", "padding": "10px", "borderRadius": "5px"})
                ]),
                dcc.Interval(id='hospital-data-refresh', interval=5_000)
            ])
        ], width=8, className="health-monitoring-panel",) ,

        # Status Panel (Unit Status)
        dbc.Col([
            html.Div("UNIT STATUS", className="panel-header"),
            html.Div(id='soldier-status-table', className="status-table"),
            html.Div([
                html.Div("SYSTEM STATUS:", className="system-status-label"),
                html.Div(id='system-status', children="ALL SYSTEMS NOMINAL",className="system-status-normal")
            ], className="system-status"),
            dcc.Interval(id='status-refresh', interval=5_000)
        ], width=4, className="status-panel")
    ]),


    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Admissions in last 24 hours", style={"textAlign": "center","backgroundColor": "#474A49"}, className="panel-header"),
                dbc.CardBody(
                    id="admissions",
                    children=[admissions_graph],
                    className="status-table"
                ),
            ])
        ])
    ]),
    # Admissions Graph
    # html.Div(admissions_graph, style={"marginTop": "20px"}),

    # Patient Priority Table
    # html.Table([
    #     html.Thead(
    #         html.Tr([html.Th("Patient"), html.Th("Age"), html.Th("Priority")])
    #     ),
    #     html.Tbody([
    #         html.Tr([html.Td(patient["id"]), html.Td(patient["age"]), html.Td(patient["priority"])])
    #         for patient in patients_info
    #     ])
    # ], style={"marginTop": "20px", "width": "100%", "border": "1px solid #DDD"}),

    # Triage Summary Section

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("TRIAGE SUMMARY", style={"color": COLORS['text'], "font-family": "Orbitron"}),
                dbc.CardBody(id="triage-summary-body-classification", children="No data available."),
                dbc.CardFooter([
                    dbc.Button("Download PDF", id="download-btn", color="success", className="mt-2"),
                    dcc.Download(id="pdf-download")
                ])
            ], className="mt-4", style={"backgroundColor": COLORS["panel"], "borderColor": COLORS["text"]})
        ], width=12)
    ]),

    # dcc.Dropdown(
    # id='soldier-selector',
    # options=[{'label': sid, 'value': sid} for sid in triage_results.keys()],
    # placeholder="Select soldier"
    #    ),
    html.Div([
        dcc.Textarea(
            id='injury-description',
            placeholder='Enter injury description...',
            style={'width': '100%', 'height': 100}
        ),
        html.Button('Generate Treatment Plan', id='generate-treatment-btn'),
        html.Div(id='treatment-plan-output', style={'whiteSpace': 'pre-line'}),
        html.Button('Save as PDF', id='save-pdf-btn', disabled=True),
        dcc.Download(id='pdf-download-1'),
        dcc.Store(id='treatment-plan-store'),  # Single download component
    ]),



    # Footer with comms status
    dbc.Row([
        dbc.Col([
            html.Div(id='comms-status', children="COMMS: STABLE",
                     className="comms-status"),
            html.Div(id='last-update', className="last-update")
        ], width=12)
    ], className="footer")
], style={'backgroundColor': COLORS['background'], 'height': '100vh'})

# Callbacks
@app.callback(
    [Output('selected-soldier-data','data'),
     Output('soldier-status-table', 'children'),
     Output('system-status', 'children'),
     Output('system-status', 'className'),
     Output('comms-status', 'children'),
     Output('last-update', 'children'),
     Output('live-clock', 'children')],
    [
     Input('status-refresh', 'n_intervals')]
)
def update_dashboard(status_intervals):
    try:
        # Check connection status
        if not opc_thread.is_connected:
            opc_thread.run()
            print("Reconnected to OPC UA server")

        # Get data from OPC UA
        objects = opc_thread.client.get_objects_node()
        soldiers = [node for node in objects.get_children()
                   if node.get_browse_name().Name.startswith("Soldier")]

        soldier_data = []
        status_rows = []
        critical_count = 0

        
            
        for soldier in soldiers:
            try:
                # Get soldier data with error handling for each field
                # name = soldier.get_browse_name().Name
                status = soldier.get_child("2:Priority").get_value()
                if status == "CRITICAL":
                    critical_count += 1

                gps_value = soldier.get_child("2:GPS").get_value()
                if ',' in gps_value:
                    lat, lon = gps_value.split(',')
                else:
                    lat, lon = "0", "0"  # Default coordinates if format is wrong

                soldier_data.append({
                    "soldier_id": soldier.get_browse_name().Name,
                    "heart_rate": soldier.get_child("2:HeartRate").get_value(),
                    "body_temp": soldier.get_child("2:BodyTemp").get_value(),
                    "latitude": float(lat.strip()),
                    "longitude": float(lon.strip()),
                    "status": status
                })
                



                # Create status table row
                status_rows.append(
                    dbc.Row([
                        dbc.Col(soldier.get_browse_name().Name, width=3),
                        dbc.Col(f"{soldier.get_child('2:HeartRate').get_value():.2f} BPM", width=3),
                        dbc.Col(f"{soldier.get_child('2:BodyTemp').get_value():.1f}°C", width=3),
                        dbc.Col(status, width=3,className=f"status-{status.lower()}"),
                        # dbc.Col(triage_results.get(name, {}).get("treatment_plan", "Unknown"), width=6)
            
                    ], className="status-row")
                )
            except Exception as e:
                print(f"Error processing {soldier.get_browse_name().Name}: {e}")
                continue

        df = pd.DataFrame(soldier_data)



        # System status logic
        if critical_count > 0:
            system_status = f"WARNING: {critical_count} CRITICAL UNITS"
            status_class = "system-status-warning"
            comms_status = "COMMS: PRIORITY TRAFFIC"
        else:
            system_status = "ALL SYSTEMS NOMINAL"
            status_class = "system-status-normal"
            comms_status = "COMMS: STABLE"

        current_time = datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y")

        return (
            soldier_data,
            status_rows,
            system_status,
            status_class,
            comms_status,
            f"LAST UPDATE: {current_time}",
            current_time.split()[0],
             # Just the time for clock
        )

    except Exception as e:
        print(f"Dashboard error: {e}")
        return (
            "Empty dataframe",
            "STATUS UNAVAILABLE",
            "SYSTEM OFFLINE",
            "system-status-offline",
            "COMMS: OFFLINE",
            "LAST UPDATE: FAILED",
            datetime.datetime.now().strftime("%H:%M:%S")
        )


#Update the Triage Summary body in the callback
@app.callback(
    Output("triage-summary-body", "children"),
    [Input("status-refresh", "n_intervals")]
)
def update_triage_summary(n):
    global triage_results , triage_classification_results
    try:
        # Get OPC UA object nodes
        objects = opc_thread.client.get_objects_node()
        soldiers = [node for node in objects.get_children()
                    if node.get_browse_name().Name.startswith("Soldier")]

        critical_count = 0

        for soldier in soldiers:
            try:
                name = soldier.get_browse_name().Name
                status = soldier.get_child("2:Status").get_value()

                if status == "CRITICAL":
                    critical_count += 1

                hr = soldier.get_child("2:HeartRate").get_value()
                rsp = soldier.get_child("2:RespiratoryRate").get_value()
                bt = soldier.get_child("2:BodyTemp").get_value()
                spo2 = soldier.get_child("2:OxygenSaturation").get_value()
                spbp = soldier.get_child("2:SystolicBP").get_value()
                dp = soldier.get_child("2:DiastolicBP").get_value()
                age = soldier.get_child("2:Age").get_value()
                gender = soldier.get_child("2:Gender").get_value()
                weight = soldier.get_child("2:Weight").get_value()
                height = soldier.get_child("2:Height").get_value()

                


                # Initialize history if not present
                if name not in history:
                    history[name] = {
                        "heart_rate": [],
                        "body_temp": [],
                        "spo2": [],
                        "systolic_bp": [],
                        "diastolic_bp": [],
                        "respiratory_rate": []
                    }

                # Append current vitals
                history[name]["heart_rate"].append(hr)
                history[name]["body_temp"].append(bt)
                history[name]["systolic_bp"].append(spbp)
                history[name]["diastolic_bp"].append(dp)
                history[name]["spo2"].append(spo2)
                history[name]["respiratory_rate"].append(rsp)

                
                # Keep only last 3 readings
                for key in history[name]:
                    history[name][key] = history[name][key][-3:]


                required_keys = ["heart_rate", "body_temp", "systolic_bp", "diastolic_bp", "spo2", "respiratory_rate"]
            


                # Run triage if enough data
            
                vitals_obj = Vitals(
                    heart_rate=history[name]["heart_rate"],
                    blood_pressure=list(zip(history[name]["systolic_bp"], history[name]["diastolic_bp"])),
                    spo2=history[name]["spo2"],
                    resp_rate=history[name]["respiratory_rate"]
                )
                request = SoldierRequest(
                    soldier_id=name,
                    vitals=vitals_obj,
                    injury_description="Gunshot wound"  # Replace with dynamic input if available
                    )
                triage = triage_system.determine_triage(request.vitals, request.injury_description)
                triage_results[name] = triage
                    # triage_classification_results[name] = triage

            except Exception as e:
                print(f"Error processing {soldier.get_browse_name().Name}: {e}")
                continue

        # Build UI summary list
        items = []
        for soldier_id, result in triage_results.items():
            items.append(html.Div([
                html.H5("Current Status",style={"color": COLORS['text']}),
                html.Div(children= f"Triage:{result['current_vitals']['category']} ({result['current_vitals']['priority']})"),
                html.H5(soldier_id, style={"color": COLORS['text']}),
                # html.P(f"Priority: {result.get('priority', 'N/A')}"),
                html.P(f"Plan: {result.get('treatment_plan', 'N/A')}"),
                html.P("This is a test entry.")
            ], className="mb-3", style={"borderBottom": "1px solid #555"}))
        
        print(f"items: {items}")
        return items

    except Exception as e:
        print(f"Unexpected error in update_triage_summary: {e}")
        return [html.P("Error fetching triage data.")]


# def collect_triage_data():
#     global triage_results, triage_classification_results
#     items = []
#     try:

#          # Get OPC UA object nodes
#         objects = opc_thread.client.get_objects_node()
#         soldiers = [node for node in objects.get_children()
#                     if node.get_browse_name().Name.startswith("Soldier")]

#         critical_count = 0

#         for soldier in soldiers:
#             try:
#                 name = soldier.get_browse_name().Name
#                 status = soldier.get_child("2:Status").get_value()

#                 if status == "CRITICAL":
#                     critical_count += 1

#                 hr = soldier.get_child("2:HeartRate").get_value()
#                 rsp = soldier.get_child("2:RespiratoryRate").get_value()
#                 bt = soldier.get_child("2:BodyTemp").get_value()
#                 spo2 = soldier.get_child("2:OxygenSaturation").get_value()
#                 spbp = soldier.get_child("2:SystolicBP").get_value()
#                 dp = soldier.get_child("2:DiastolicBP").get_value()
#                 age = soldier.get_child("2:Age").get_value()
#                 gender = soldier.get_child("2:Gender").get_value()
#                 weight = soldier.get_child("2:Weight").get_value()
#                 height = soldier.get_child("2:Height").get_value()

                


#                 # Initialize history if not present
#                 if name not in history:
#                     history[name] = {
#                         "heart_rate": [],
#                         "body_temp": [],
#                         "spo2": [],
#                         "systolic_bp": [],
#                         "diastolic_bp": [],
#                         "respiratory_rate": []
#                     }

#                 # Append current vitals
#                 history[name]["heart_rate"].append(hr)
#                 history[name]["body_temp"].append(bt)
#                 history[name]["systolic_bp"].append(spbp)
#                 history[name]["diastolic_bp"].append(dp)
#                 history[name]["spo2"].append(spo2)
#                 history[name]["respiratory_rate"].append(rsp)

                
#                 # Keep only last 3 readings
#                 for key in history[name]:
#                     history[name][key] = history[name][key][-3:]


#                 required_keys = ["heart_rate", "body_temp", "systolic_bp", "diastolic_bp", "spo2", "respiratory_rate"]
            


#                 # Run triage if enough data
#                 if all(len(history[name][k]) == 3 for k in required_keys):
#                     vitals_obj = Vitals(
#                         heart_rate=history[name]["heart_rate"],
#                         blood_pressure=list(zip(history[name]["systolic_bp"], history[name]["diastolic_bp"])),
#                         spo2=history[name]["spo2"],
#                         resp_rate=history[name]["respiratory_rate"]
#                     )
#                     request = SoldierRequest(
#                         soldier_id=name,
#                         vitals=vitals_obj,
#                         injury_description="Gunshot wound"  # Replace with dynamic input if available
#                     )
#                     triage = triage_system.determine_triage(request.vitals, request.injury_description)
#                     triage_results[name] = triage
#                     # triage_classification_results[name] = triage

#             except Exception as e:
#                 print(f"Error processing {soldier.get_browse_name().Name}: {e}")
#                 continue


        
        
        
        # Build UI summary list
    #     items = []
    #     for soldier_id, result in triage_results.items():
    #         items.append(html.Div([
    #             html.H5("Current Status", style={"color": COLORS['text']}),
    #             html.Div(children=f"Triage:{result['current_vitals']['category']} ({result['current_vitals']['priority']})"),
    #             html.H5(soldier_id, style={"color": COLORS['text']}),
    #             html.P(f"Plan: {result.get('treatment_plan', 'N/A')}"),
    #             html.P("This is a test entry.")
    #         ], className="mb-3", style={"borderBottom": "1px solid #555"}))
        
    #     return items, triage_results

    # except Exception as e:
    #     print(f"Unexpected error in collect_triage_data: {e}")
    #     return [html.P("Error fetching triage data.")], {}

@app.callback(
    Output("hospital-health-status", "children"),
    Input("hospital-data-refresh", "n_intervals")
)
def update_hospital_patient_data(n):
    try:
        # Sample dummy data — replace with OPC UA or database source
        patients = [
            {"id": "Patient_001", "HR": 82, "Temp": 37.2, "SpO2": 97, "image_url": "./assets/RESQ.jpg"},
            {"id": "Patient_002", "HR": 95, "Temp": 38.1, "SpO2": 93, "image_url": "./assets/patient_001.jpg"}
        ]


        components = []
        for patient in patients:
            components.append(html.Div([
                html.Div([
                    html.Img(src=app.get_asset_url("Patient1.png"),style={"width": "100px","height": "100px","borderRadius": "50%","objectFit": "cover"}),
                    html.Div([
                        html.H5(patient["id"], style={"color": COLORS["text"]}),
                        html.P(f"Heart Rate: {patient['HR']} BPM"),
                        html.P(f"Temperature: {patient['Temp']}°C"),
                        html.P(f"Oxygen Saturation: {patient['SpO2']}%"),
                    ], style={"display": "inline-block", "margin-left": "15px"})
                ], style={"display": "flex", "align-items": "center"}),
                html.Hr()
            ]))
        return components
    except Exception as e:
        return html.Div(f"Error fetching hospital data: {e}")




@app.callback(
        Output("triage-summary-body-classification", "children"),
        [Input("status-refresh", "n_intervals")]
)

def update_triage_classification(n):
    global triage_classification_results
    try:
        # Get OPC UA object nodes
        objects = opc_thread.client.get_objects_node()
        soldiers = [node for node in objects.get_children()
                    if node.get_browse_name().Name.startswith("Soldier")]

        critical_count = 0

        for soldier in soldiers:
            try:
                name = soldier.get_browse_name().Name
                status = soldier.get_child("2:Status").get_value()

                if status == "CRITICAL":
                    critical_count += 1

                hr = soldier.get_child("2:HeartRate").get_value()
                rsp = soldier.get_child("2:RespiratoryRate").get_value()
                bt = soldier.get_child("2:BodyTemp").get_value()
                spo2 = soldier.get_child("2:OxygenSaturation").get_value()
                spbp = soldier.get_child("2:SystolicBP").get_value()
                dp = soldier.get_child("2:DiastolicBP").get_value()
                age = soldier.get_child("2:Age").get_value()
                gender = soldier.get_child("2:Gender").get_value()
                weight = soldier.get_child("2:Weight").get_value()
                height = soldier.get_child("2:Height").get_value()

                
                if name not in class_name:
                    class_name[name] = {
                        "heart_rate": [],
                        "respiratory_rate": [],
                        "body_temp": [],
                        "spo2": [],
                        "systolic_bp": [],
                        "diastolic_bp": [],
                        "age" :[],
                        "gender": [],
                        "weight": [],
                        "height": []
                        
                    }

                # For filling the values for the classification model
                class_name[name]["heart_rate"].append(hr)
                class_name[name]["respiratory_rate"].append(rsp)
                class_name[name]["body_temp"].append(bt)
                class_name[name]["spo2"].append(spo2)
                class_name[name]["systolic_bp"].append(spbp)
                class_name[name]["diastolic_bp"].append(dp)
                class_name[name]["age"].append(age)
                class_name[name]["gender"].append(gender)
                class_name[name]["weight"].append(weight)
                class_name[name]["height"].append(height)
            
                
                # vitals_obj_classification = Vitals_Classification(
                #     heart_rate =  class_name[name]['heart_rate'],
                #     respiratory_rate =  class_name[name]['respiratory_rate'],
                #     body_temp =  class_name[name]['body_temp'],
                #     spo2 =  class_name[name]['spo2'],
                #     sbp =  class_name[name]['systolic_bp'],
                #     dbp =  class_name[name]['diastolic_bp'],
                #     age =  class_name[name]['age'],
                #     gender=  class_name[name]['gender'],
                #     weight =  class_name[name]['weight'],
                #     height =  class_name[name]['height'], 
                # )

                all_vitals = [
                    Vitals_Classification(
                        heart_rate=[99], respiratory_rate=[16], body_temp=[36.65], spo2=[95.01],
                        sbp=[118], dbp=[72], age=[41], gender=[0], weight=[96.00], height=[1.83]
                    ),
                    Vitals_Classification(
                        heart_rate=[83], respiratory_rate=[12], body_temp=[36.04], spo2=[98.58],
                        sbp=[111], dbp=[84], age=[50], gender=[0], weight=[79.30], height=[1.67]
                    ),
                    Vitals_Classification(
                        heart_rate=[79], respiratory_rate=[12], body_temp=[36.88], spo2=[95.98],
                        sbp=[130], dbp=[70], age=[22], gender=[1], weight=[79.87], height=[1.92]
                    ),
                    Vitals_Classification(
                        heart_rate=[66], respiratory_rate=[15], body_temp=[36.95], spo2=[97.91],
                        sbp=[131], dbp=[77], age=[61], gender=[1], weight=[53.92], height=[1.89]
                    ),
                    Vitals_Classification(
                        heart_rate=[72], respiratory_rate=[16], body_temp=[36.80], spo2=[98.00],
                        sbp=[120], dbp=[80], age=[20], gender=[1], weight=[78.00], height=[1.78]
                    )
                ]

                # test_vitals = Vitals_Classification(
                #     heart_rate=[99],
                #     respiratory_rate=[16],
                #     body_temp=[36.65],
                #     spo2=[95.01],
                #     sbp=[118],
                #     dbp=[72],
                #     age=[41],
                #     gender=[0],  # assuming 0 = male, 1 = female
                #     weight=[96.00],
                #     height=[1.83]
                # )


                # request_obj_cls = SoldierRequest_classify(
                #     soldier_id = name,
                #     vitals = vitals_obj_classification,
                # )

                # test_request = SoldierRequest_classify(
                #     soldier_id="Test_Soldier_1",
                #     vitals=test_vitals
                # )

                # label, prob, reasoning_text, raw_scores = triage_system.predict_triage_category(request_obj_cls.vitals)
                # triage_classification_results [name] = triage_classification


                # label, prob, reasoning_text, raw_scores = triage_system.predict_triage_category(test_request.vitals)
                # triage_classification_results[name] = {
                #     "label": label,
                #     "confidence": prob,
                #     "reasoning": reasoning_text,
                #     "raw_scores": raw_scores
                # }


                # vitals_obj_classification = Vitals_Classification(
                #     heart_rate=[hr],
                #     respiratory_rate=[rsp],
                #     body_temp=[bt],
                #     spo2=[spo2],
                #     sbp=[spbp],
                #     dbp=[dp],
                #     age=[age],
                #     gender=[gender],
                #     weight=[weight],
                #     height=[height],
                # )

                # label, prob, reasoning_text, raw_scores = triage_system.predict_triage_category(vitals_obj_classification)

                # triage_classification_results[name] = {
                #     "label": label,
                #     "confidence": prob,
                #     "reasoning": reasoning_text,
                #     "raw_scores": raw_scores
                # }

                for i, vitals in enumerate(all_vitals, start=1):
                    label, prob, reasoning_text, raw_scores = triage_system.predict_triage_category(vitals)
                    triage_classification_results[f"soldier_{i}"] = {
                        "label": label,
                        "confidence": prob,
                        "reasoning": reasoning_text,
                        "raw_scores": raw_scores
                    }
                # Log results
                logging.info("Here I am logging the info")
                logging.info(f"[{name}] Prediction Results:")
                logging.info(f"    Label: {label}")
                logging.info(f"    Probability: {prob}")
                logging.info(f"    Reasoning: {reasoning_text}")
                logging.info(f"    Raw Scores: {raw_scores}")

                # Optional: Log input vitals
                logging.info(f"    Input Vitals - HR: {hr}, RR: {rsp}, BT: {bt}, SPO2: {spo2}, SBP: {spbp}, DBP: {dp}, AGE: {age}, GENDER: {gender}, WEIGHT: {weight}, HEIGHT: {height}")
            except Exception as e:
                print(f"Error processing {soldier.get_browse_name().Name}: {e}")
                continue

        # Build UI summary list
        items_classification = []
        for soldier_id, result in triage_classification_results.items():
            alert_indicator = None
            report_button = None

            if result["label"] == "CRITICAL":
                alert_indicator = html.Span(className="red-dot blink")

                # Blinking button to generate report
                report_button = html.Button(
                    "Generate Report",
                    id={'type': 'generate-report-btn', 'index': soldier_id},  # Use pattern matching callback if needed
                    className="blink",
                    style={'marginTop': '10px', 'backgroundColor': 'red', 'color': 'white'}
                )

            items_classification.append(html.Div([
                html.H5(alert_indicator if alert_indicator else "","Classification Prediction", style={"color": COLORS['text']}),
                html.H5(soldier_id, style={"color": COLORS['text']}),
                html.P(f"Predicted Triage Category: {result['label']}"),
                html.P(f"Confidence: {result['confidence']}"),
                html.P(f"Reasoning: {result['reasoning']}"),
                html.P(f"Raw Scores: {result['raw_scores']}"),
                report_button if report_button else None
            ], className="mb-3", style={"borderBottom": "1px solid #555"}))
        
        print(f"items: {items_classification}")
        return items_classification

    except Exception as e:
        print(f"Unexpected error in update_triage_summary: {e}")
        return [html.P("Error fetching triage data.")]

import requests
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



treatment_plan_store = {}

# Modified callbacks
@app.callback(
    [Output('treatment-plan-output', 'children'),
     Output('save-pdf-btn', 'disabled'),
     Output('treatment-plan-store', 'data')],  # Store the data
    Input('generate-treatment-btn', 'n_clicks'),
    State('injury-description', 'value')
)
def generate_treatment_plan(n_clicks, injury_description):
    if n_clicks is None or not injury_description:
        raise PreventUpdate
    
    # Create dummy vitals
    current_vitals = {'hr': 90, 'sbp': 120, 'dbp': 80, 'spo2': 98, 'rr': 16}
    predicted_vitals = {'hr': 88, 'sbp': 118, 'dbp': 78, 'spo2': 97, 'rr': 15}
    
    treatment_prompt = (
        f"Generate a battlefield treatment plan for:\n"
        f"- Current vitals: HR {current_vitals['hr']}, BP {current_vitals['sbp']}/{current_vitals['dbp']}, "
        f"SpO2 {current_vitals['spo2']}%, RR {current_vitals['rr']}\n"
        f"- Predicted vitals: HR {predicted_vitals['hr']}, BP {predicted_vitals['sbp']}/{predicted_vitals['dbp']}, "
        f"SpO2 {predicted_vitals['spo2']}%, RR {predicted_vitals['rr']}\n"
        f"- Injury: {injury_description}\n\n"
        "Provide a concise step-by-step treatment plan considering battlefield constraints."
    )
    
    treatment_plan = get_ollama_response(
        treatment_prompt,
        temperature=0.7,
        max_tokens=300,
        model_name="llama3"
    )
    
    # Return both the UI update and the stored data
    return treatment_plan, False, {'plan': treatment_plan, 'injury': injury_description}

@app.callback(
    Output('pdf-download-1', 'data'),
    Input('save-pdf-btn', 'n_clicks'),
    State('treatment-plan-store', 'data'),
    prevent_initial_call=True
)
def save_treatment_pdf(n_clicks, stored_data):
    print(f"PDF Button clicked! Clicks: {n_clicks}")  # Debug 1
    print(f"Stored data: {stored_data}")  # Debug 2
    
    if n_clicks is None or not stored_data:
        print("Preventing update - missing data or no clicks")  # Debug 3
        raise PreventUpdate
    
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Battlefield Treatment Plan", ln=1, align='C')
        pdf.ln(10)
        
        # Content
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=stored_data['plan'])
        
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        print("PDF generated successfully!")  # Debug 4
        return dcc.send_bytes(pdf_bytes, "treatment_plan.pdf")
        
    except Exception as e:
        print(f"PDF generation failed: {str(e)}")  # Debug 5
        raise PreventUpdate

@app.callback(
    Output("pdf-download", "data"),
    Input("download-btn", "n_clicks"),
    prevent_initial_call=True
)
def download_pdf(n_clicks):
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Triage Summary", ln=True, align='C')

    for soldier_id, result in triage_classification_results.items():
        pdf.ln(5)
        pdf.cell(200, 10, txt=f"Soldier: {soldier_id}", ln=True)
        pdf.cell(200, 10, txt=f"Priority: {result.get('Label', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Plan: {result.get('treatment_plan', 'N/A')}", ln=True)

    # Get the PDF as a bytes string
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return dcc.send_bytes(pdf_bytes, "triage_summary.pdf")


def clean_shutdown():
    try:
        opc_thread.disconnect()
        print("OPCUA client is disconnected")
    except Exception as e:
        print(f"Error disconnecting OPC Client {e}")

def find_free_port(min_port=8050, max_port=9000, max_tries=100):
    """Find a free port using random sampling"""
    for _ in range(max_tries):
        port = random.randint(min_port, max_port)
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            try:
                sock.bind(('', port))
                sock.listen(1)
                return port
            except socket.error:
                continue
    raise ValueError(f"No free ports found after {max_tries} attempts")

if __name__ == '__main__':
    timeout = 10
    waited = 0
    while not opc_thread.is_connected and waited < timeout:
        time.sleep(0.5)
        waited += 0.5

    try:
        free_port = find_free_port()
        print(free_port)
        url = f"http://localhost:{free_port}"
        # print(f"🚀 Dashboard is starting on: {url}")

        # Open the dashboard in the default web browser
        webbrowser.open(url)
        app.run(port=free_port, debug=True)

        

    except KeyboardInterrupt:
        print("Server is stopped by the user")

    finally:
        clean_shutdown()
        print("Server is now stopped")



