import os
from huggingface_hub import HfApi, login
def upload():
    # 1. Load your token securely from Kaggle Secrets or environment variables
    # Make sure you generated a NEW token after that last leak!
    hf_token = os.environ.get("HF_TOKEN") 
    login(token=hf_token)

    api = HfApi()
    target_repo = "naren-1219/orion-170m" 

    # 2. Create the repo (does nothing if it already exists)
    api.create_repo(repo_id=target_repo, repo_type="model", exist_ok=True)

    print(f"Uploading files to {target_repo}...")

    # 3. Upload using upload_folder (fastest and cleanest way)
    api.upload_folder(
        folder_path=".", 
        # Only grabs your final PyTorch weights, config, and model scripts
        allow_patterns=["orion_final.pt", "config.json", "*.py"], 
        repo_id=target_repo,
        repo_type="model"
    )

    print(f"Done! Your model is live: https://huggingface.co/{target_repo}")