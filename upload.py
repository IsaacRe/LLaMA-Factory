from huggingface_hub import HfApi, login
import os
import glob

hf_token = os.getenv("HF_TOKEN")

try:
    login(token=hf_token)
    print("Successfully logged in to Hugging Face.")
except Exception as e:
    print(f"Failed to log in to Hugging Face. Please ensure you have a valid token set (e.g., as an HF_TOKEN environment variable) or run `huggingface-cli login` in your terminal. Error: {e}")
    exit()

api = HfApi()

base_checkpoints_dir = "./checkpoints/OpenThoughts--Llama-3.1-8B-Instruct/"

repo_id = "isaacrehg/OpenThoughts-Llama-3.1-8B-Instruct"

print(f"Checking for checkpoint folders in: {base_checkpoints_dir}")

checkpoint_folders = sorted(glob.glob(os.path.join(base_checkpoints_dir, "checkpoint-*")))

if not checkpoint_folders:
    print(f"No checkpoint folders found in '{base_checkpoints_dir}'. Please verify the path.")
else:
    print(f"Found checkpoint folders: {', '.join(checkpoint_folders)}")
    for checkpoint_path in checkpoint_folders:
        checkpoint_folder_name = os.path.basename(checkpoint_path)
        branch_name = checkpoint_folder_name

        print(f"\n--- Processing '{checkpoint_folder_name}' ---")
        print(f"Attempting to upload '{checkpoint_path}' to '{repo_id}' on branch '{branch_name}'...")

        api.create_branch(repo_id=repo_id, branch=branch_name, repo_type="model", exist_ok=True)
        api.upload_large_folder(
            folder_path=checkpoint_path,
            repo_id=repo_id,
            repo_type="model",
            revision=branch_name,
            print_report_every=10000,
            #commit_message=f"Upload of {checkpoint_folder_name}"
        )
        print(f"Successfully uploaded '{checkpoint_folder_name}' to '{repo_id}' on branch '{branch_name}'!")

