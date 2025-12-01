import subprocess
import yaml
import os
import sys

def run_command(command):
    print(f"Running: {command}")
    # Use sys.executable to ensure we use the same python interpreter
    if command.startswith("python "):
        command = f"{sys.executable} {command[7:]}"
        
    process = subprocess.Popen(command, shell=True)
    process.wait()
    if process.returncode != 0:
        raise Exception(f"Command failed: {command}")

def main():
    config_path = 'data/config.yaml'
    
    # Stage 1: Initial Training
    print("Starting Stage 1: Initial Training...")
    try:
        run_command("python main.py train --config data/config.yaml")
    except Exception as e:
        print(f"Stage 1 failed: {e}")
        sys.exit(1)
    
    # Modify config for GAN
    print("Modifying config for GAN training...")
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Try to preserve comments with string replacement
        if 'use_gan: false' in content:
            new_content = content.replace('use_gan: false', 'use_gan: true')
            with open(config_path, 'w') as f:
                f.write(new_content)
        elif 'use_gan: False' in content:
            new_content = content.replace('use_gan: False', 'use_gan: true')
            with open(config_path, 'w') as f:
                f.write(new_content)
        else:
            # Fallback to yaml parser
            print("String 'use_gan: false' not found, using yaml parser (comments may be lost)...")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            config['model']['use_gan'] = True
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
                
    except Exception as e:
        print(f"Failed to modify config: {e}")
        sys.exit(1)

    # Stage 2: GAN Training
    # Read config to get output_dir
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    output_dir = config.get('data', {}).get('output_dir', './output')
    best_model_path = os.path.join(output_dir, 'best_model.pt')
    
    if not os.path.exists(best_model_path):
        print(f"Error: Best model not found at {best_model_path}")
        sys.exit(1)
        
    print(f"Starting Stage 2: GAN Training resuming from {best_model_path}...")
    try:
        run_command(f"python main.py train --config data/config.yaml --resume {best_model_path}")
    except Exception as e:
        print(f"Stage 2 failed: {e}")
        sys.exit(1)

    print("All training stages completed successfully.")

if __name__ == "__main__":
    main()
