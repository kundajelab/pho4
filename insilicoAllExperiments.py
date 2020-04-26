import itertools
import subprocess

class CyclicalRepeatPatterns:
    def __init__(self):
        self.data = []
        self.extensions = []

    def generateAllCyclicalPermutations(self, token):
        ret = []
        for idx in range(len(token)):
            ret.append(token[-idx:]+token[:-idx])
        return ret

    def insertToken(self, token, length):
        allPermutations = self.generateAllCyclicalPermutations(token)
        for permute in allPermutations:
            if permute in self.data or permute in self.extensions:
                return
        self.data.append(token)
        for idx in range(1, int(length/len(token))+1):
            self.extensions.append(token*idx)

    def generateAllPatternsOfLength(self, length=4):
        chars = "ACGT"
        for N in range(1, length+1):
            for item in itertools.product(chars, repeat=N):
                self.insertToken("".join(item), length)

    def getData(self):
        return self.data

p = CyclicalRepeatPatterns()
p.generateAllPatternsOfLength(6)
repeatPatterns=p.getData()

pbexo_pho4 = {
    "--gpus": "0,1,2",
    "--model": "data/models/pho4_pbexo_model.h5",
    "--out_pred_len": "200",
    "--bed": "data/pho4_pbexo/pho4.pbexo.bed",
    "--out_dir": "data/insilico_exp/pbexo_pho4/"
}
pbexo_cbf1 = {
    "--gpus": "0,1,2",
    "--model": "data/models/cbf1_pbexo_model.h5",
    "--out_pred_len": "200",
    "--bed": "data/cbf1_pbexo/cbf1.pbexo.bed",
    "--out_dir": "data/insilico_exp/pbexo_cbf1/"
}
chipexo_cbf1 = {
    "--gpus": "0,1,2",
    "--model": "data/models/cbf1_chipexo_model.h5",
    "--out_pred_len": "225",
    "--bed": "data/cbf1_chipexo/cbf1.chipexo.bed",
    "--out_dir": "data/insilico_exp/chipexo_cbf1/"
}

tasks = [pbexo_pho4, pbexo_cbf1, chipexo_cbf1]
for pattern in repeatPatterns:
    for task in tasks:
        command = ['python','insilicoExperimentPipeline.py','--repeat', pattern]
        for key in task:
            command.append(key)
            if key == "--out_dir":
                command.append(task[key]+pattern+"/")
            else: command.append(task[key])
        print(command)
        subprocess.check_call(command)
