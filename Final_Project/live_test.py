import joblib
import pandas as pd
from scapy.all import sniff, IP, ICMP, TCP
import numpy as np
import time

# --- MODEL FILES AND SETUP ---
MODEL_FILE = 'multiclass_ids_model.joblib'
INTERFACE = 'eth0' 
CLASS_LABELS = {0: "Normal Traffic", 1: "ICMP Flood Attack", 2: "SYN Flood Attack"}

# Features list (must match training order exactly)
FEATURE_COLUMNS = [
    'frame.time_delta', 'ip.proto', 'ip.len', 
    'tcp.srcport', 'tcp.dstport', 'tcp.flags.syn', 
    'icmp.type', 'icmp.seq', 'icmp.code'
]
# --- END SETUP ---


# Load the trained model
try:
    model = joblib.load(MODEL_FILE)
    print("IDS Multiclass Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: '{MODEL_FILE}' not found. Please run ai_model.py first.")
    exit()

# Variable to track time between packets
last_time = time.time() 


def extract_features(packet):
    global last_time
    
    current_time = packet.time
    time_delta = current_time - last_time
    last_time = current_time
    
    # 1. Extract IP features
    if IP in packet:
        ip_proto = packet[IP].proto
        ip_len = packet[IP].len
    else:
        ip_proto = 0
        ip_len = 0
    
    # 2. Extract TCP features
    if TCP in packet:
        tcp_srcport = packet[TCP].sport
        tcp_dstport = packet[TCP].dport
        tcp_flags_syn = 1 if 'S' in str(packet[TCP].flags) else 0
    else:
        tcp_srcport = 0
        tcp_dstport = 0
        tcp_flags_syn = 0
        
    # 3. Extract ICMP features
    if ICMP in packet:
        icmp_type = packet[ICMP].type
        icmp_code = packet[ICMP].code
        icmp_seq = packet[ICMP].seq
    else:
        icmp_type = 0
        icmp_code = 0
        icmp_seq = 0
        
    # Compile features in the EXACT ORDER of training
    features_list = [
        time_delta,       
        ip_proto,         
        ip_len,           
        tcp_srcport,      
        tcp_dstport,      
        tcp_flags_syn,    
        icmp_type,        
        icmp_seq,         
        icmp_code         
    ]
    
    return features_list


def packet_callback(packet):
    # Only process packets with an IP layer
    if IP in packet:
        try:
            # Extract features
            features = extract_features(packet)
            
            # Convert features into a single-row DataFrame for the model
            df_predict = pd.DataFrame([features], columns=FEATURE_COLUMNS)
            
            # Make prediction
            prediction = model.predict(df_predict)[0]
            
            source_ip = packet[IP].src
            
            result_label = CLASS_LABELS[prediction]
            
            if prediction != 0:
                # Output detected attack with its class (1 or 2)
                print(f"[{source_ip}] -> [*** DETECTED {prediction} ***]: {result_label}")
            else:
                # Output safe traffic
                print(f"[{source_ip}] -> [SAFE]: {result_label}")
        
        except Exception as e:
            # Pass silently if a rare packet cannot be processed (to maintain IDS speed)
            pass 


# 3. Start Live Monitoring
print("\n--- Starting Live IDS Monitor (Multiclass) ---")
print(f"Listening on interface: {INTERFACE}. Classes: 0, 1, 2.")
print("Press Ctrl+C to stop.")

# Begin sniffing
sniff(iface=INTERFACE, prn=packet_callback, store=0)
