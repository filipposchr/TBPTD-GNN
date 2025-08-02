import argparse
import subprocess

datasets = [ "edit-mathoverflow",  "edit-facebook_wall","edit-topology" ,  "edit-mlwikiquote", "edit-plwikiquote", "edit-digg_reply", "edit-SMS", "edit-rt-pol", "edit-slashdot_reply","edit-wamazon", "edit-mgwikipedia", "edit-tgwiktionary", "edit-ltwiktionary"]
#datasets = [ "edit-infectious","edit-facebook_wall" ]
# Argument parser
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--bet", choices=["sh", "sfm"], help="Choose betweenness mode: 'sh' or 'sfm'")
group.add_argument("--close", choices=["f", "sh"], help="Choose closeness mode: 'f' or 'sh'")

args = parser.parse_args()

if args.bet is not None:
    mode_flag = "--bet"
    mode_value = args.bet
else:
    mode_flag = "--close"
    mode_value = args.close

for d in datasets:
    print(f"\n - Running evaluation for dataset: *{d}* -")
    command = ["python", "-u", "main.py", "-d", d, mode_flag, mode_value, "--test"]
    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"Error: Evaluation failed for dataset {d}")
