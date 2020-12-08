# Processing AR R5 SELEX-seq data
options(java.parameters = "-Xmx200000M")

library(rJava)
library(SELEX)
library(SelexGLM)
library(grid)

workDir = "/users/amr1/pho4/SelexGLM/processing/AR_R5"
dir.create(file.path(workDir), showWarnings = FALSE, recursive = TRUE)
selex.config(workingDir=workDir, maxThreadNumber=800)

selexDir = "/users/amr1/pho4/SelexGLM"
processedDataDir = "/oak/stanford/groups/akundaje/amr1/pho4/selexglm"
saveDir = "/results/AR_R5"
dir.create(file.path(selexDir, saveDir), showWarnings = FALSE, recursive = TRUE)

selex.defineSample('R0',
                   paste(processedDataDir, "/gr_selex/r0/SRR5340724.1_1.fastq", sep = ""),
                   'R0',
                   0, 23, '', 'TGGAA')

selex.defineSample('AR_R5',
                   paste(processedDataDir, "/ar_selex/r5/SRR5340728.1_1.fastq", sep = ""),
                   'AR_R5',
                   5, 23, '', 'TGGAA')


r0 = selex.sample(seqName = 'R0', sampleName='R0', round = 0)
r0.split = selex.split(r0)
r0.train = r0.split$train
r0.test = r0.split$test
dataSample = selex.sample(seqName = 'AR_R5', sampleName = 'AR_R5', round = 5)

# MARKOV MODELS
kmax = selex.kmax(sample = r0.test,threshold=100)

mm_transit = selex.mm(sample = r0.train, order = NA, crossValidationSample =r0.test, Kmax = kmax, mmMethod = "TRANSITION")
data.kmerTableT = selex.affinities(sample = dataSample, k = 23, minCount = 1, markovModel = mm_transit)
write.csv(x=data.kmerTableT, file=file.path(selexDir, saveDir,"/kmerTableTransit.csv"))
data.symmetricKmerTableT = getKmerCountAffinities(dataSample, k = 23, minCount = 1, markovModel = mm_transit)
write.csv(x=data.symmetricKmerTableT, file=file.path(selexDir, saveDir,"/symmetricKmerTableTransit.csv"))

mm_divide = selex.mm(sample = r0.train, order = NA, crossValidationSample =r0.test, Kmax = kmax, mmMethod = "DIVISION")
data.kmerTableD = selex.affinities(sample = dataSample, k = 23, minCount = 1, markovModel = mm_divide)
write.csv(x=data.kmerTableD, file=file.path(selexDir, saveDir,"/kmerTableDivide.csv"))
data.symmetricKmerTableD = getKmerCountAffinities(dataSample, k = 23, minCount = 1, markovModel = mm_divide)
write.csv(x=data.symmetricKmerTableD, file=file.path(selexDir, saveDir,"/symmetricKmerTableDivide.csv"))