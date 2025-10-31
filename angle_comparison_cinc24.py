import yaml
import os

if __name__=="__main__":
    USE_DATASETS = ["MMPD"]  # ["KISMED","UBFC-rPPG","PURE","RLAP","COHFACE","VIPL-HR-V1","MMPD"] 
    for dataset in USE_DATASETS:
        cfg_file =f"configs/cinc24/angle_comparison/{dataset}_30.yaml"
        with open(cfg_file, 'r') as f:
            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        for angle in [15,20,25,30,35,40,45,50,60,70,90]:
            yaml_cfg["UNSUPERVISED"]["DATA"]["PREPROCESS"]["ROI_SEGMENTATION"]["THRESHOLD"]=angle
            yaml_cfg["UNSUPERVISED"]["DATA"]["DO_PREPROCESS"]=False
            cfg_out = f"configs/cinc24/angle_comparison/{dataset}_temp.yaml"
            with open(cfg_out,'w') as f:
                yaml.dump(yaml_cfg,f,)
            
            os.system(f"conda run -n rppg-toolbox python main.py --config_file {cfg_out}")