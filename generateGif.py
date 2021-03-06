import os
import subprocess

variation = "distance"

for task in os.listdir("insilico_exp/"):
    if task == ".DS_Store": continue
    for repeat in os.listdir("insilico_exp/"+task+"/"+variation):
        if repeat == ".DS_Store": continue
        for filename in os.listdir("insilico_exp/"+task+"/"+variation+"/"+repeat):
            if filename.endswith(".zip"):
                subprocess.check_call(["tar", "xvf", "insilico_exp/"+task+"/"+variation+"/"+repeat+"/"+filename])
                subprocess.check_call(["convert",
                    "data/insilico_exp/"+task+"/"+variation+"/"+repeat+"/"+filename.split('.')[0]+"/*.png",
                    filename.split('.')[0]+".gif"])
                subprocess.check_call(["mv",
                    filename.split('.')[0]+".gif",
                    "insilico_exp/"+task+"/"+variation+"/"+repeat+"/"])
                subprocess.check_call(["rm", "-rf", "data"])
                subprocess.check_call(["rm", "-rf", "insilico_exp/"+task+"/"+variation+"/"+repeat+"/"+filename])
