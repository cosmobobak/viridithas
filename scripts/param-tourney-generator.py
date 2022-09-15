
folder_path = input("Enter the folder path: ")
n_values = input("Enter the number of values: ")
n_values = int(n_values)
param_files = []
for i in range(n_values):
    f_name = input(f"Enter the name of file {i+1}: ")
    param_files.append(f_name)
n_rounds = input("Enter the number of rounds: ")
n_rounds = int(n_rounds)

engine_names = [f"p_{f}" for f in param_files]
param_files = [folder_path + "/" + f for f in param_files]

command = "nice -n 10 cutechess-cli "
for fname, ename in zip(param_files, engine_names):
    command += f"-engine cmd=virtue/target/release/viridithas arg=\"--eparams\" arg=\"{fname}\" name=\"{ename}\" "

command += "-engine cmd=virtue/target/release/viridithas name=dev "
command += f"-each timemargin=400 proto=uci tc=100/8+0.08 -rounds {n_rounds} -concurrency 60 -openings file=uhobook.pgn format=pgn -repeat -games 2"

print(command)
