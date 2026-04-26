import json

# Fix the parameter shadowing in tilted_loss.ipynb
filepath = 'tilted_loss.ipynb'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# We replace the specific strings that cause the issue
content = content.replace('def local_cost(p):', 'def local_cost(opt_params):')
content = content.replace('pr = qaoa_circuit(p, p=p)', 'pr = qaoa_circuit(opt_params, p=p)')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Notebook fixed successfully!")
