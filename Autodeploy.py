import shutil
import os

def self_deploy():
    # Specify the destination directory for deployment
    deployment_directory = "/path/to/deployment/directory"

    # Create the deployment directory if it doesn't exist
    if not os.path.exists(deployment_directory):
        os.makedirs(deployment_directory)

    # Copy the necessary files to the deployment directory
    script_file = "intrusion_detection.py"
    dataset_file = "data.csv"

    shutil.copy(script_file, deployment_directory)
    shutil.copy(dataset_file, deployment_directory)

    # Print deployment success message
    print("Self-deployment completed successfully in:", deployment_directory)

# Example usage
self_deploy()
