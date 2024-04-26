import pandas as pd
import numpy as np

# Define the number of rows for your dataset
num_rows = 1000

# Generate random data for each column
data = {
    'duration': np.random.randint(1, 1000, num_rows),
    'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], num_rows),
    'service': np.random.choice(['http', 'ftp', 'smtp'], num_rows),
    'flag': np.random.choice(['SF', 'S0', 'REJ'], num_rows),
    'src_bytes': np.random.randint(0, 10000, num_rows),
    'dst_bytes': np.random.randint(0, 10000, num_rows),
    'land': np.random.randint(0, 2, num_rows),
    'wrong_fragment': np.random.randint(0, 5, num_rows),
    'urgent': np.random.randint(0, 3, num_rows),
    'hot': np.random.randint(0, 10, num_rows),
    'num_failed_logins': np.random.randint(0, 5, num_rows),
    'logged_in': np.random.randint(0, 2, num_rows),
    'num_compromised': np.random.randint(0, 5, num_rows),
    'root_shell': np.random.randint(0, 2, num_rows),
    'su_attempted': np.random.randint(0, 2, num_rows),
    'num_root': np.random.randint(0, 5, num_rows),
    'num_file_creations': np.random.randint(0, 5, num_rows),
    'num_shells': np.random.randint(0, 2, num_rows),
    'num_access_files': np.random.randint(0, 2, num_rows),
    'num_outbound_cmds': np.random.randint(0, 2, num_rows),
    'is_host_login': np.random.randint(0, 2, num_rows),
    'is_guest_login': np.random.randint(0, 2, num_rows),
    'count': np.random.randint(0, 100, num_rows),
    'srv_count': np.random.randint(0, 100, num_rows),
    'serror_rate': np.random.uniform(0, 1, num_rows),
    'srv_serror_rate': np.random.uniform(0, 1, num_rows),
    'rerror_rate': np.random.uniform(0, 1, num_rows),
    'srv_rerror_rate': np.random.uniform(0, 1, num_rows),
    'same_srv_rate': np.random.uniform(0, 1, num_rows),
    'diff_srv_rate': np.random.uniform(0, 1, num_rows),
    'srv_diff_host_rate': np.random.uniform(0, 1, num_rows),
    'dst_host_count': np.random.randint(0, 100, num_rows),
    'dst_host_srv_count': np.random.randint(0, 100, num_rows),
    'dst_host_same_srv_rate': np.random.uniform(0, 1, num_rows),
    'dst_host_diff_srv_rate': np.random.uniform(0, 1, num_rows),
    'dst_host_same_src_port_rate': np.random.uniform(0, 1, num_rows),
    'dst_host_srv_diff_host_rate': np.random.uniform(0, 1, num_rows),
    'dst_host_serror_rate': np.random.uniform(0, 1, num_rows),
    'dst_host_srv_serror_rate': np.random.uniform(0, 1, num_rows),
    'dst_host_rerror_rate': np.random.uniform(0, 1, num_rows),
    'dst_host_srv_rerror_rate': np.random.uniform(0, 1, num_rows)
}

# Create a DataFrame from the generated data
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('Gtest_data.csv', index=False)

# Print the first few rows of the DataFrame
print(df.head())
