import os
import subprocess

for task in os.listdir("insilico_exp/"):
    if task == ".DS_Store": continue
    for repeat in os.listdir("insilico_exp/"+task):
        if repeat == ".DS_Store": continue
        for filename in os.listdir("insilico_exp/"+task+"/"+repeat):
            if filename.endswith(".zip"):
                subprocess.check_call(["tar", "xvf", "insilico_exp/"+task+"/"+repeat+"/"+filename])
                subprocess.check_call(["convert",
                    "data/insilico_exp/"+task+"/"+repeat+"/"+filename.split('.')[0]+"/*.png",
                    filename.split('.')[0]+".gif"])
                subprocess.check_call(["mv",
                    filename.split('.')[0]+".gif",
                    "insilico_exp/"+task+"/"+repeat+"/"])
                subprocess.check_call(["rm", "-rf", "data"])
                subprocess.check_call(["rm", "-rf", "insilico_exp/"+task+"/"+repeat+"/"+filename])