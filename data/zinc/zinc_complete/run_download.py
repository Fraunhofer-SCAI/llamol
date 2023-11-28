import concurrent.futures
import subprocess

shell_file = "download_zinc.sh"
num_parallel = 8

def execute_command(command):
    print("Running: ", command)
    subprocess.run(command, shell=True)

commands = []
with open(shell_file, "r") as file:
    for line in file:
        line = line.strip()
        if line.startswith("mkdir") and "wget" in line:
            commands.append(line)

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(execute_command, commands, chunksize=num_parallel)

print("Downloads completed")