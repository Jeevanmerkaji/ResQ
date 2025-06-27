from opcua import Client
import random
import time
import threading
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class OPCUAClient(threading.Thread):
    def __init__(self, url="opc.tcp://localhost:8051", timeout=100000):
        super().__init__()
        self.url = url
        self.timeout = timeout
        self.client = Client(self.url, timeout=self.timeout)
        self.is_connected = False
        self.stop_event = threading.Event()
        self.soldier_data = []  # Store soldier data in a class attribute

    def run(self):
        try:
            # Connect to the OPC UA server
            self.client.connect()
            self.is_connected = True
            logging.info("OPC UA Connected Successfully")

            # Get the objects node from the server
            objects = self.client.get_objects_node()
            soldiers = [node for node in objects.get_children()
                        if node.get_browse_name().Name.startswith("Soldier")]
            
            while not self.stop_event.is_set():
                soldier_data = []
                for soldier in soldiers:
                    try:
                        soldier_id = soldier.get_browse_name().Name
                        status = soldier.get_child("2:Status").get_value()
                        gps_value = soldier.get_child("2:GPS").get_value()
                        if ',' in gps_value:
                            lat, lon = gps_value.split(',')
                        else:
                            lat, lon = "0", "0"

                        soldier_data.append({
                            "soldier_id": soldier_id,
                            "heart_rate": soldier.get_child("2:HeartRate").get_value(),
                            "respiratory_rate": soldier.get_child("2:RespiratoryRate").get_value(),
                            "body_temp": soldier.get_child("2:BodyTemp").get_value(),
                            "systolic_bp": soldier.get_child("2:SystolicBP").get_value(),
                            "diastolic_bp": soldier.get_child("2:DiastolicBP").get_value(),
                            "latitude": float(lat.strip()),
                            "longitude": float(lon.strip()),
                            "status": status,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "spo2": soldier.get_child("2:OxygenSaturation").get_value(),
                            "age": soldier.get_child("2:Age").get_value(),
                            "gender" : soldier.get_child("2:Gender").get_value(),
                            "weight" : soldier.get_child("2:Weight").get_value(),
                            "height": soldier.get_child("2:Height").get_value(),
                            "hrv" : soldier.get_child("2:Derived_HRV").get_value(),
                        })
                    except Exception as e:
                        logging.error(f"Error processing soldier {soldier.get_browse_name().Name}: {e}")
                        continue

                self.soldier_data = soldier_data  # Store the collected data
                time.sleep(1)  # Update every second

        except Exception as e:
            logging.error(f"Error connecting to OPC UA server: {e}")

    def fetch_soldier_details(self, soldier_id):
        """Fetch detailed data for a specific soldier."""
        if not self.is_connected:
            raise Exception("Client is not connected to the server")
        
        # Check if soldier data exists
        for soldier in self.soldier_data:
            if soldier["soldier_id"] == soldier_id:
                # Simulating fetching more detailed data
                hr = soldier["heart_rate"]
                temp = soldier["body_temp"]
                status = soldier["status"]
                gps = f"{soldier['latitude']},{soldier['longitude']}"
                # bp = f"{random.randint(90, 180)}/{random.randint(60, 120)}"
                resp_rate = soldier["respiratory_rate"]
                # spo2 = f"{random.randint(95, 100)}%"
                spo2 = soldier['spo2']
                systolic_bp = soldier['systolic_bp']
                diastolic_bp = soldier['diastolic_bp']
                movement = "MOVING" if random.random() > 0.5 else "STATIC"
                
                return {
                    "soldier_id": soldier_id,
                    "heart_rate": hr,
                    "body_temp": temp,
                    "status": status,
                    "gps": gps,
                    "bp": resp_rate,
                    "spo2": spo2,
                    "movement": movement,
                    "systolic_bp" : systolic_bp,
                    "diastolic_bp" : diastolic_bp
                }

        return None  # If no soldier found with that ID

    def stop(self):
        """Stop the thread safely."""
        self.stop_event.set()
        if self.client:
            self.client.disconnect()
            logging.info("OPC UA Client disconnected")
