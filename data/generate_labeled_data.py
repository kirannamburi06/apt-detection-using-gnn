import pandas as pd
import random
from datetime import datetime, timedelta


def generate_benign_sequence():
    """Generate normal activity logs"""
    logs = []
    start_time = datetime(2024, 2, 1, 9, 0, 0)

    benign_processes = ['chrome.exe', 'outlook.exe', 'winword.exe', 'excel.exe']
    benign_files = ['document.docx', 'report.xlsx', 'presentation.pptx']
    benign_actions = ['READ', 'WRITE', 'OPEN']
    benign_ips = ['google.com', 'office365.com', '8.8.8.8']

    for i in range(20):
        timestamp = start_time + timedelta(minutes=i * 15)
        process = random.choice(benign_processes)

        if random.random() > 0.5:
            obj = random.choice(benign_files)
        else:
            obj = random.choice(benign_ips)

        action = random.choice(benign_actions)

        logs.append({
            'timestamp': timestamp,
            'process': process,
            'action': action,
            'object': obj,
            'pid': random.randint(1000, 5000)
        })

    return logs, 0  # Label 0 = benign


def generate_attack_sequence():
    """Generate APT attack logs"""
    logs = []
    start_time = datetime(2024, 2, 1, 9, 0, 0)

    # Stage 1: Phishing (0-1 hour)
    logs.append({
        'timestamp': start_time + timedelta(minutes=5),
        'process': 'outlook.exe',
        'action': 'DOWNLOAD',
        'object': 'attachment.docx',
        'pid': 1234
    })

    logs.append({
        'timestamp': start_time + timedelta(minutes=10),
        'process': 'winword.exe',
        'action': 'EXECUTE',
        'object': 'macro.vbs',
        'pid': 1235
    })

    # Stage 2: Persistence (1-2 hours)
    logs.append({
        'timestamp': start_time + timedelta(minutes=70),
        'process': 'powershell.exe',
        'action': 'DOWNLOAD',
        'object': 'malware.exe',
        'pid': 2345
    })

    logs.append({
        'timestamp': start_time + timedelta(minutes=75),
        'process': 'powershell.exe',
        'action': 'WRITE',
        'object': 'registry_key',
        'pid': 2345
    })

    # Stage 3: Reconnaissance (2-3 hours)
    logs.append({
        'timestamp': start_time + timedelta(minutes=140),
        'process': 'malware.exe',
        'action': 'CONNECT',
        'object': '10.0.0.0/24',
        'pid': 3456
    })

    logs.append({
        'timestamp': start_time + timedelta(minutes=145),
        'process': 'malware.exe',
        'action': 'READ',
        'object': 'user_list.txt',
        'pid': 3456
    })

    # Stage 4: Credential theft (3-4 hours)
    logs.append({
        'timestamp': start_time + timedelta(minutes=200),
        'process': 'mimikatz.exe',
        'action': 'READ',
        'object': 'lsass.exe',
        'pid': 4567
    })

    logs.append({
        'timestamp': start_time + timedelta(minutes=205),
        'process': 'mimikatz.exe',
        'action': 'WRITE',
        'object': 'credentials.txt',
        'pid': 4567
    })

    # Stage 5: Exfiltration (4-5 hours)
    logs.append({
        'timestamp': start_time + timedelta(minutes=260),
        'process': 'powershell.exe',
        'action': 'CONNECT',
        'object': '203.0.113.5:443',
        'pid': 5678
    })

    logs.append({
        'timestamp': start_time + timedelta(minutes=265),
        'process': 'powershell.exe',
        'action': 'UPLOAD',
        'object': 'credentials.txt',
        'pid': 5678
    })

    logs.append({
        'timestamp': start_time + timedelta(minutes=270),
        'process': 'powershell.exe',
        'action': 'DELETE',
        'object': 'malware.exe',
        'pid': 5678
    })

    return logs, 1  # Label 1 = attack


def generate_dataset(num_benign=10, num_attacks=10):
    """Generate complete labeled dataset"""
    dataset = []

    print(f"Generating {num_benign} benign sequences...")
    for i in range(num_benign):
        logs, label = generate_benign_sequence()
        dataset.append({
            'sequence_id': f'benign_{i}',
            'logs': logs,
            'label': label
        })

    print(f"Generating {num_attacks} attack sequences...")
    for i in range(num_attacks):
        logs, label = generate_attack_sequence()
        dataset.append({
            'sequence_id': f'attack_{i}',
            'logs': logs,
            'label': label
        })

    # Shuffle
    random.shuffle(dataset)

    # Save
    all_logs = []
    labels_file = []

    for seq in dataset:
        seq_id = seq['sequence_id']
        label = seq['label']

        for log in seq['logs']:
            log['sequence_id'] = seq_id
            all_logs.append(log)

        labels_file.append({'sequence_id': seq_id, 'label': label})

    # Save logs
    df_logs = pd.DataFrame(all_logs)
    df_logs.to_csv('./labeled_audit_logs.csv', index=False)

    # Save labels
    df_labels = pd.DataFrame(labels_file)
    df_labels.to_csv('./sequence_labels.csv', index=False)

    print(f"\n✅ Generated dataset:")
    print(f"   Total logs: {len(all_logs)}")
    print(f"   Total sequences: {len(dataset)}")
    print(f"   Benign: {num_benign}, Attacks: {num_attacks}")
    print(f"   Saved to: data/labeled_audit_logs.csv")
    print(f"   Labels: data/sequence_labels.csv")


if __name__ == "__main__":
    generate_dataset(num_benign=10, num_attacks=10)