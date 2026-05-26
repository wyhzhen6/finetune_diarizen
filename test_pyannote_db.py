import os
from pyannote.database import registry

# Load the custom database configuration
os.environ["PYANNOTE_DATABASE_CONFIG"] = "database.yml"

try:
    # Get the protocol
    protocol = registry.get_protocol("CustomData.SpeakerDiarization.Finetune")
    
    # Iterate over the train set
    print("Testing train set...")
    train_files = list(protocol.train())
    print(f"Found {len(train_files)} files in train set.")
    
    if len(train_files) > 0:
        first_file = train_files[0]
        print(f"First file URI: {first_file['uri']}")
        print(f"First file audio path: {first_file['audio']}")
        print(f"First file annotation:\n{first_file['annotation']}")
        
    print("Database loading successful!")
except Exception as e:
    print(f"Error loading database: {e}")
    import traceback
    traceback.print_exc()
