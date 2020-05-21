import os
import subprocess

for task in os.listdir("noGif/"):
    if task == "pbexo_pho4": taskname = "0-"+task
    elif task == "pbexo_cbf1": taskname = "1-"+task
    elif task == "chipexo_cbf1": taskname = "2-"+task
    else: continue
    for repeat in os.listdir("noGif/"+task+"/"):
        if repeat == ".DS_Store": continue
        for filename in os.listdir("noGif/"+task+"/"+repeat):
            if "left" in filename: filename2 = "0-"+filename.split('.')[0]
            elif "right" in filename: filename2 = "1-"+filename.split('.')[0]
            elif "both_mirror" in filename: filename2 = "3-"+filename.split('.')[0]
            elif "both" in filename: filename2 = "2-"+filename.split('.')[0]
            else: continue
            subprocess.check_call(["cp", "noGif/"+task+"/"+repeat+"/"+filename,
                "all/"+filename2+"-"+repeat+"-"+taskname+".png"])