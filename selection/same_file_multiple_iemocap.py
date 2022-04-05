import os
import sys
import subprocess

aug_file = sys.argv[1]
outdir = sys.argv[2]
names = [str(x) for x in range(20)]
for name in names : 
    bashCmd=["bash", "file_pipeline_iemocap.sh", os.path.join(outdir,name),
             aug_file]
    process = subprocess.Popen(bashCmd, stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output)
    
