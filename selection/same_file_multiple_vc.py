import os
import sys
import subprocess
names = [str(x) for x in range(10)]
aug_file = sys.argv[1]

outdir = sys.argv[2]
for name in names : 
    bashCmd=["bash", "file_vcpipeline.sh",os.path.join(outdir,name), aug_file ]

    process = subprocess.Popen(bashCmd, stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output)
    
