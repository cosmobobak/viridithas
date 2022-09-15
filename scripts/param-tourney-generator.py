
folder_path = input("Enter the folder path: ")
n_values = input("Enter the number of values: ")
n_values = int(n_values)
automatic_file_generation = input("Do you want to generate the files automatically? (y/n): ")
param_files = []
if automatic_file_generation == "n":
    for i in range(n_values):
        f_name = input(f"Enter the name of file {i+1}: ")
        param_files.append(f_name)
elif automatic_file_generation == "y":
    for i in range(n_values):
        # pad with zeroes to make it three wide
        idx = i + 1
        idx = str(idx).zfill(3)
        f_name = f"localsearch{idx}.params"
        param_files.append(f_name)
else:
    print("Invalid input")
    exit()
n_rounds = input("Enter the number of rounds: ")
n_rounds = int(n_rounds)

engine_names = [f"p_{f}" for f in param_files]
param_files = [folder_path + "/" + f for f in param_files]

command = "nice -n 10 cutechess-cli "
for fname, ename in zip(param_files, engine_names):
    command += f"-engine cmd=target/release/viridithas arg=\"--eparams\" arg=\"{fname}\" name=\"{ename}\" "

command += "-engine cmd=target/release/viridithas name=dev "
command += f"-each timemargin=400 proto=uci tc=100/8+0.08 -concurrency 60 -openings file=uhobook.pgn format=pgn -repeat -games 2 -rounds {n_rounds} -pgnout tune-comparison.pgn"

print(command)
