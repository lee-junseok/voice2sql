import sys
import json
import argparse
sys.path.append("./src")
from annotate_question import annotate_question
from evaluate_question import main

parser = argparse.ArgumentParser()
parser.add_argument('-c','-config_path', default = "config/model_config.json",
	help='Path to the config file.')
parser.add_argument('-q','--question', required = True,
	help='Natural language question for the database.')
args = parser.parse_args()

with open(args.config_path, 'r') as f:
    con = json.load(f)

anno_filename, headers = annotate_question(
	args.question, con["table_id"], con["dir_in"], con["dir_out"], "test")

args = ['-model_path', con["model"], '-data_path', con["data_path"], \
        "-anno_data_path", con["anno_path"]]
sql_code, result_list = main(anno_filename, headers, args)

print()
print("SQL code:", sql_code)
print()
print("Execution result:", ', '.join([str(x) for x in result_list]))
print()
