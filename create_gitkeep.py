import os

# Dossiers à créer avec .gitkeep
directories = [
    'data/raw',
    'data/processed',
    'data/examples',
    'data/temp',
    'models/embeddings',
    'models/graph',
    'models/text',
    'checkpoints',
    'results',
    'logs',
    'tests',
    'docs/images',
    'notebooks/examples'
]

base_path = 'C:/Projects/GraphRAG'

for directory in directories:
    full_path = os.path.join(base_path, directory)
    os.makedirs(full_path, exist_ok=True)
    
    gitkeep_path = os.path.join(full_path, '.gitkeep')
    
    # Créer .gitkeep si n'existe pas
    if not os.path.exists(gitkeep_path):
        with open(gitkeep_path, 'w') as f:
            f.write(f"# Keep {directory} directory\n")
        print(f"✓ Created .gitkeep in {directory}")
    else:
        print(f"○ .gitkeep already exists in {directory}")

print("\n✓✓✓ All .gitkeep files created!")