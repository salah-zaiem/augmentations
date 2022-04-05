import os
import sys
import subprocess
names = [str(x) for x in range(50)]
for name in names : 
    bashCmd=["bash", "whole_pipelinevc.sh", "vctest_dir/"+name]

    process = subprocess.Popen(bashCmd, stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output)
    
