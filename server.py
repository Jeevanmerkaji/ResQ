from opcua import Server
import random
import time
from datetime import datetime

# Create and configure the server
server = Server()
server.set_endpoint("opc.tcp://localhost:8051")
ns = server.register_namespace("SOLDIER_DATA")
server.set_server_name("Combat Medical OPC UA Server")

# Create patient (soldier) nodes
soldiers = []
for soldier_id in range(1, 5):
    soldier = server.nodes.objects.add_object(ns, f"Soldier{soldier_id}")

    # Add variables for each health metric
    soldiers.append({
        "id": soldier.add_variable(ns, "PatientID", soldier_id),
        "heart_rate": soldier.add_variable(ns, "HeartRate", 0),
        "resp_rate": soldier.add_variable(ns, "RespiratoryRate", 0),
        "timestamp": soldier.add_variable(ns, "Timestamp", ""),
        "body_temp": soldier.add_variable(ns, "BodyTemp", 0.0),
        "location": soldier.add_variable(ns, "GPS", "0.0"),
        "spo2": soldier.add_variable(ns, "OxygenSaturation", 0.0),
        "systolic_bp": soldier.add_variable(ns, "SystolicBP", 0),
        "status": soldier.add_variable(ns, "Status" ,"OK"),
        "diastolic_bp": soldier.add_variable(ns, "DiastolicBP", 0),
        "age": soldier.add_variable(ns, "Age", random.randint(18, 44)),  # Age < 45
        "gender": soldier.add_variable(ns, "Gender", random.choice(["Male", "Female"])),
        "weight": soldier.add_variable(ns, "Weight", round(random.uniform(50.0, 100.0), 1)),
        "height": soldier.add_variable(ns, "Height", round(random.uniform(1.5, 2.0), 2)),
        "hrv": soldier.add_variable(ns, "Derived_HRV", 0.0),
        "priority": soldier.add_variable(ns, "Priority", "GREEN")
    })


def triage_rules(hr, sbp, dbp, spo2, rr):
    if sbp < 90 and hr > 130:
        return "RED", "Immediate"
    elif spo2 < 92 or rr > 24:
        return "YELLOW", "Delayed"
    elif hr < 40 or spo2 < 80:
        return "BLACK", "Expectant"
    else:
        return "GREEN", "Minor"

# Start server
server.start()
print("OPC UA Server started at opc.tcp://localhost:8051")

try:
    while True:
        for soldier in soldiers:
            # Simulate vitals
            # hr = random.randint(60, 100)
            # rr = random.randint(12, 20)
            # temp = round(random.uniform(36.0, 37.5), 1)
            # spo2 = round(random.uniform(95.0, 100.0), 1)
            # sbp = random.randint(110, 140)
            # dbp = random.randint(70, 90)
            # hrv = round(random.uniform(20.0, 120.0), 2)
            # color, priority = triage_rules(hr, sbp, dbp, spo2, rr)
            if random.random() < 0.25:
                hr = random.choice([35, 140])
                rr = random.choice([10, 26])
                spo2 = random.choice([78.5, 90.0])
                sbp = random.choice([85, 135])
            else:
                hr = random.randint(60, 100)
                rr = random.randint(12, 20)
                spo2 = round(random.uniform(95.0, 100.0), 1)
                sbp = random.randint(110, 140)

            # These should always be defined regardless of branch
            dbp = random.randint(70, 90)
            temp = round(random.uniform(36.0, 37.5), 1)
            hrv = round(random.uniform(20.0, 120.0), 2)

                    # Determine triage
            color, priority = triage_rules(hr, sbp, dbp, spo2, rr)

            # Set simulated values
            soldier["heart_rate"].set_value(hr)
            soldier["resp_rate"].set_value(rr)
            soldier["body_temp"].set_value(temp)
            soldier["spo2"].set_value(spo2)
            soldier["systolic_bp"].set_value(sbp)
            soldier["diastolic_bp"].set_value(dbp)
            soldier["hrv"].set_value(hrv)
            soldier['location'].set_value(f"{random.uniform(-90,90):.4f},{random.uniform(-180,180):.4f}")
            soldier["priority"].set_value(priority)
            

            # Timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            soldier["timestamp"].set_value(timestamp)

            print(f"[{timestamp}] Soldier ID {soldier['id'].get_value()} - HR: {hr}, RR: {rr}, Temp: {temp}, SpO2: {spo2}, BP: {sbp}/{dbp}, HRV: {hrv}, Priority: {priority}")

        time.sleep(1)

except Exception as e:
    print("Server error:", e)

finally:
    server.stop()
    print("Server stopped")
