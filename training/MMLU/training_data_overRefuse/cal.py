import json

with open('./OverRefuseDataset.json', 'r') as f:
    data = json.load(f)
    num_instances = len(data['instances'])
    print(f"文件中共有 {num_instances} 个instances")