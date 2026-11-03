"""Embed recorded data in the literal interactive report template."""
import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
base = Path(__file__).parent
data = json.loads((base / 'results/hierarchy.json').read_text(encoding='utf-8'))
fields = ('name', 'bytes', 'stage1_nodes', 'stage2_nodes', 'bytes_per_stage1_node',
          'stage1_nodes_per_stage2_node', 'bytes_per_stage2_node')
with (base / 'results/rates.csv').open('w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(data['samples'])
controlled = [sample for sample in data['samples'] if sample['name'].startswith('redundancy_')]
minimal = {'samples': [{key: sample[key] for key in (*fields, 'stage1_spans', 'stage2_spans')}
                       for sample in (controlled or data['samples'])]}
embedded = json.dumps(minimal, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
template = (base / 'report-template.html').read_text(encoding='utf-8')
args.output.write_text(template.replace('__HNET_DATA__', embedded), encoding='utf-8')
print(args.output)
