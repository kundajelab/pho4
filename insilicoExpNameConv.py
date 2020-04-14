import optparse
import os

parser = optparse.OptionParser()

parser.add_option('--dir',
    action="store", dest="dir",
    help="which folder", default=None)

options, args = parser.parse_args()

for filename in os.listdir(options.dir):
    if len(filename.split('.')[0]) == 1:
        os.rename(options.dir+filename, options.dir+"00"+filename)
    elif len(filename.split('.')[0]) == 2:
        os.rename(options.dir+filename, options.dir+"0"+filename)

print("DONE")