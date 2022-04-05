## Selecting and parametrizing augmentations

Apart from the audio files, this folder contains all what is needed to select and parametrize a suitable distribution for VoxForge or VoxCeleb. The simplest way to use this, is to change the wav directories  in the files "file_pipeline_voxforge.sh" and "file_vcpipeline.sh" and launch : 

```
python launch_voxforge.py $namedirout
```
or 

```
python launch_vc.py $namedirout
```

Given an example of distribution parametrization "example.json", one can compute its score regarding speaker identification, through launching :

```
bash file_vcpipeline.sh example.json
```

