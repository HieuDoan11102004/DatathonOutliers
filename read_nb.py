import sys
sys.stdout.reconfigure(encoding='utf-8')
import json

nb = json.load(open('baseline.ipynb', 'r', encoding='utf-8'))
for c in nb['cells']:
    print(f"--- {c['cell_type']} ---")
    if isinstance(c['source'], list):
        print(''.join(c['source']))
    else:
        print(c['source'])
    print()
