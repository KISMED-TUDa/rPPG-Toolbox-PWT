import yaml
import os

if __name__=="__main__":
    # cfg_file = "configs/cinc24/angle_comparison/KISMED_UNSUPERVISED_30.yaml"
    # with open(cfg_file, 'r') as f:
    #     yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    # for angle in [15,20,25,30,35,40,45,50,60,70,90]:
    #     yaml_cfg["UNSUPERVISED"]["DATA"]["PREPROCESS"]["ROI_SEGMENTATION"]["THRESHOLD"]=angle
    #     yaml_cfg["UNSUPERVISED"]["DATA"]["DO_PREPROCESS"]=False
    #     cfg_out = "configs/cinc24/angle_comparison/KISMED_UNSUPERVISED_temp.yaml"
    #     with open(cfg_out,'w') as f:
    #         yaml.dump(yaml_cfg,f,)
        
    #     os.system(f"conda run -n rppg-toolbox python main.py --config_file {cfg_out}")
        
    cfg_file ="configs/cinc24/angle_comparison/UBFC-rPPG_30.yaml"
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    for angle in [15,20,25,30,35,40,45,50,60,70,90]:
        yaml_cfg["UNSUPERVISED"]["DATA"]["PREPROCESS"]["ROI_SEGMENTATION"]["THRESHOLD"]=angle
        yaml_cfg["UNSUPERVISED"]["DATA"]["DO_PREPROCESS"]=True
        cfg_out = "configs/cinc24/angle_comparison/UBFC-rPPG_UNSUPERVISED_temp.yaml"
        with open(cfg_out,'w') as f:
            yaml.dump(yaml_cfg,f,)
        
        os.system(f"conda run -n rppg-toolbox python main.py --config_file {cfg_out}")