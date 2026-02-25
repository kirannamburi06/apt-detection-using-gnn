import pandas as pd
import random
from datetime import datetime, timedelta


def generate_toy_data(num_samples=100, output_file='./toy_audit_logs.csv'):
    """Generate synthetic audit logs for testing"""

    processes = ['chrome.exe', 'outlook.exe', 'powershell.exe', 'cmd.exe',
                 'winword.exe', 'explorer.exe', 'notepad.exe', 'svchost.exe']

    files = ['config.txt', 'document.docx', 'passwords.txt', 'system.dll',
             'update.exe', 'malware.exe', 'credentials.dat', 'important.pdf']

    actions = ['READ', 'WRITE', 'EXECUTE', 'DELETE', 'CONNECT', 'DOWNLOAD']

    ips = ['192.168.1.100', '10.0.0.50', '203.0.113.5', '8.8.8.8',
           'google.com', 'malicious.com']

    logs = []
    start_time = datetime(2024, 2, 1, 9, 0, 0)

    for i in range(num_samples):
        timestamp = start_time + timedelta(minutes=i * 3)
        process = random.choice(processes)

        if random.random() > 0.5:
            obj = random.choice(files)
        else:
            obj = random.choice(ips)

        action = random.choice(actions)

        logs.append({
            'timestamp': timestamp,
            'process': process,
            'action': action,
            'object': obj,
            'pid': random.randint(1000, 9999)
        })

    df = pd.DataFrame(logs)
    df.to_csv(output_file, index=False)
    print(f"✅ Generated {num_samples} log entries in {output_file}")
    return df


if __name__ == "__main__":
    generate_toy_data(100)