import argparse
from pathlib import Path
import json
from os.path import join

import constants
import test_utils



parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="RITE-test")
parser.add_argument("-p", "--predictions_path", type=str, default=None)
parser.add_argument("-t", "--test", type=str, default="mav")
parser.add_argument("-n", "--n_paths", type=int, default=0)
parser.add_argument("--force", action="store_true")
parser.add_argument("--recompute", action="store_true")
parser.add_argument("--data_path", type=str, default="_Evaluation_Data")
parser.add_argument("--results_dir", type=str, default="__results")
parser.add_argument("--pixels", type=str, default="intersection", choices=["both", "intersection", "all"])
parser.add_argument("-v", "--version", type=str, default="rrwnet")
args = parser.parse_args()


data_path = Path(args.data_path)

results_dir = Path(args.results_dir)
results_dir.mkdir(exist_ok=True)

if args.dataset == "RITE-test":
    dataset = constants.rite_test_dataset
    dataset.paths = {
        'gt_hu',
        'masks',
    }
    dataset.gt = 'gt_hu'
    if args.predictions_path is None:
        dataset.paths = dataset.paths.union({
            
            #'RITE_RollingUNetbase3',
            'BCE_base'

            # 'gt2_qureshi',
            # 'rrwnet',
            # 'RollingUNetminmax', 'RollingUNetbase',  'RollingUNetcl'
            # 'github_rrwnet_RITE_1', 'github_rrwnet_RITE_refinement'
            # 'nnUNet'
            # 'maskSPcl30', 'maskSPcl20'
            # 'maskSPcl100','maskSPcl10'
            # 'minmax1.3',
            # 'AttUNet_minmax_0.01', 'AttUNet_minmax_0.1', 'AttUNet_minmax_1.0', 'AttUNet_minmax_0.05', 'AttUNet_minmax_0.5', 
            # 'AttUNet_minmax_0.01_spcl_0.1_nseg_20', 'AttUNet_minmax_0.1_spcl_0.1_nseg_20', 'AttUNet_minmax_1.0_spcl_0.1_nseg_20', 'AttUNet_minmax_0.05_spcl_0.1_nseg_20', 'AttUNet_minmax_0.5_spcl_0.1_nseg_20',
            # 'RITE_Iternet_minmax_0.01','RITE_Iternet_minmax_0.05','RITE_Iternet_minmax_0.1','RITE_Iternet_minmax_0.5','RITE_Iternet_minmax_1.0',
            # 'RITE_Iternet_minmax_1.3', 'RITE_Iternet_minmax_1.5',
            # 'rrwnetminmax_1.0', 'rrwnetminmax_0.1', 'rrwnetminmax_0.01', 'rrwnetminmax_0.05', 'rrwnetminmax_0.5', 
            # 'RITE_UNet_minmax_0.1', 'RITE_UNet_minmax_0.5', 'RITE_UNet_minmax_0.05', 'RITE_UNet_minmax_0.01', 'RITE_UNet_minmax_1.0'
            # 'RITE_UNet++_minmax_0.1', 'RITE_UNet++_minmax_0.01', 'RITE_UNet++_minmax_0.05', 'RITE_UNet++_minmax_0.5', 'RITE_UNet++_minmax_1.0
            # 'RITE_RollingUNet_minmax_0.1', 'RITE_RollingUNet_minmax_0.5', 'RITE_RollingUNet_minmax_0.01', 'RITE_RollingUNet_minmax_0.05', 'RITE_RollingUNet_minmax_1.0', 
            # 'RITE_RRWNet_minmax_0.1_spcl_0.05', 'RITE_RRWNet_minmax_0.1_spcl_0.07',
            #'RITE_UNet_minmax_0.05_spcl_0.01','RITE_UNet_minmax_0.05_spcl_0.03','RITE_UNet_minmax_0.05_spcl_0.05','RITE_UNet_minmax_0.05_spcl_0.07','RITE_UNet_minmax_0.05_spcl_0.09',
            #'RITE_UNet++_minmax_1.0_spcl_0.01', 'RITE_UNet++_minmax_1.0_spcl_0.03','RITE_UNet++_minmax_1.0_spcl_0.05','RITE_UNet++_minmax_1.0_spcl_0.07','RITE_UNet++_minmax_1.0_spcl_0.09',
            # 'RITE_Iternet_minmax_1.3_spcl_0.01','RITE_Iternet_minmax_1.3_spcl_0.03','RITE_Iternet_minmax_1.3_spcl_0.05','RITE_Iternet_minmax_1.3_spcl_0.07','RITE_Iternet_minmax_1.3_spcl_0.09',
            #'RITE_AttUNet_minmax_0.1_spcl_0.01', 'RITE_AttUNet_minmax_0.1_spcl_0.03','RITE_AttUNet_minmax_0.1_spcl_0.05','RITE_AttUNet_minmax_0.1_spcl_0.07','RITE_AttUNet_minmax_0.1_spcl_0.09',
            # 'RITE_CTFNet_minmax_0.01', 'RITE_CTFNet_minmax_0.05', 'RITE_CTFNet_minmax_0.1', 'RITE_CTFNet_minmax_0.5', 'RITE_CTFNet_minmax_1.0'
            # 'RITE_RRWNetfinal_minmax_0.2', 'RITE_RRWNetfinal_minmax_0.4', 'RITE_RRWNetfinal_minmax_0.06', 'RITE_RRWNetfinal_minmax_0.3', 'RITE_RRWNetfinal_minmax_0.07', 'RITE_RRWNetfinal_minmax_0.09', 'RITE_RRWNetfinal_minmax_0.08', 
            # 'RITE_AttUNet_minmax_0.1_spcl_0.001', 'RITE_AttUNet_minmax_0.1_spcl_0.003', 'RITE_AttUNet_minmax_0.1_spcl_0.005', 'RITE_AttUNet_minmax_0.1_spcl_0.007', 'RITE_AttUNet_minmax_0.1_spcl_0.009', 
            # 'RITE_UNet_minmax_0.05_spcl_0.001', 'RITE_UNet_minmax_0.05_spcl_0.003', 'RITE_UNet_minmax_0.05_spcl_0.005', 'RITE_UNet_minmax_0.05_spcl_0.007', 'RITE_UNet_minmax_0.05_spcl_0.009',
            #'RITE_CTFnet_minmax_0.5_spcl_0.001', 'RITE_CTFnet_minmax_0.5_spcl_0.003', 'RITE_CTFnet_minmax_0.5_spcl_0.005', 'RITE_CTFnet_minmax_0.5_spcl_0.007', 'RITE_CTFnet_minmax_0.5_spcl_0.009',
            #'RITE_Iternet_minmax_1.3_spcl_0.001','RITE_Iternet_minmax_1.3_spcl_0.003','RITE_Iternet_minmax_1.3_spcl_0.005','RITE_Iternet_minmax_1.3_spcl_0.007','RITE_Iternet_minmax_1.3_spcl_0.009'
            #'RITE_rrwnetminmax_0.082','RITE_rrwnetminmax_0.084','RITE_rrwnetminmax_0.086','RITE_rrwnetminmax_0.088',
            #'RITE_Iternet_minmax_1.3_spcl_0.006_nseg_20', 'RITE_Iternet_minmax_1.3_spcl_0.0055_nseg_20','RITE_Iternet_minmax_1.3_spcl_0.0065_nseg_20',
            #'RITE_UNet_minmax_0.05_spcl_0.0035_nseg_20','RITE_UNet_minmax_0.05_spcl_0.0040_nseg_20','RITE_UNet_minmax_0.05_spcl_0.0045_nseg_20',
            # 'RITE_Iternet_minmax_1.3_spcl_0.0055_nseg_20','RITE_Iternet_minmax_1.3_spcl_0.006_nseg_20','RITE_Iternet_minmax_1.3_spcl_0.0065_nseg_20',
            # 'RITE_UNet_minmax_0.05_spcl_0.01', 'RITE_UNet_minmax_0.05_spcl_0.03', 'RITE_UNet_minmax_0.05_spcl_0.05', 'RITE_UNet_minmax_0.05_spcl_0.07', 'RITE_UNet_minmax_0.05_spcl_0.09', 
            # 'RITE_Iternet_minmax_1.3_spcl_0.01_nseg_20', 'RITE_Iternet_minmax_1.3_spcl_0.03_nseg_20', 'RITE_Iternet_minmax_1.3_spcl_0.05_nseg_20', 'RITE_Iternet_minmax_1.3_spcl_0.07_nseg_20', 'RITE_Iternet_minmax_1.3_spcl_0.09_nseg_20', 
            # 'RITE_Iternet_minmax_1.3_spcl_0.0051', 'RITE_Iternet_minmax_1.3_spcl_0.0053', 'RITE_Iternet_minmax_1.3_spcl_0.0057', 'RITE_Iternet_minmax_1.3_spcl_0.0059', 
            # 'RITE_Iternet_minmax_1.3_spcl_0.0061', 'RITE_Iternet_minmax_1.3_spcl_0.0063', 'RITE_Iternet_minmax_1.3_spcl_0.0067', 'RITE_Iternet_minmax_1.3_spcl_0.0069', 
            # 'RITE_UNet_minmax_0.05_spcl_0.0011', 'RITE_UNet_minmax_0.05_spcl_0.0013', 'RITE_UNet_minmax_0.05_spcl_0.0015', 'RITE_UNet_minmax_0.05_spcl_0.0017', 'RITE_UNet_minmax_0.05_spcl_0.0019', 
            # 'RITE_UNet_minmax_0.05_spcl_0.0021',  'RITE_UNet_minmax_0.05_spcl_0.0023', 'RITE_UNet_minmax_0.05_spcl_0.0025', 'RITE_UNet_minmax_0.05_spcl_0.0027', 'RITE_UNet_minmax_0.05_spcl_0.0029',
            # 'RITE_final_rrwnetminmax_0.0081', 'RITE_final_rrwnetminmax_0.0083', 'RITE_final_rrwnetminmax_0.0085', 'RITE_final_rrwnetminmax_0.0087', 'RITE_final_rrwnetminmax_0.0089', 
            # 'RITE_final_rrwnetminmax_0.12', 'RITE_final_rrwnetminmax_0.13', 'RITE_final_rrwnetminmax_0.16', 'RITE_final_rrwnetminmax_0.17', 
            # "RITE_AVcasNet_minmax0.01_SpCL0.09"
            # 'RITE_final_rrwnetminmax_0.19_spcl_0.001', 'RITE_final_rrwnetminmax_0.19_spcl_0.003', 'RITE_final_rrwnetminmax_0.19_spcl_0.005', 'RITE_final_rrwnetminmax_0.19_spcl_0.007', 'RITE_final_rrwnetminmax_0.19_spcl_0.009', 
            # 'RITE_final_rrwnetminmax_0.19_spcl_0.01', 'RITE_final_rrwnetminmax_0.19_spcl_0.03', 'RITE_final_rrwnetminmax_0.19_spcl_0.05', 'RITE_final_rrwnetminmax_0.19_spcl_0.07', 'RITE_final_rrwnetminmax_0.19_spcl_0.09', 
            # 'RITE_AVcasNet_minmax0.01_SpCL0.0075', 'RITE_AVcasNet_minmax0.01_SpCL0.008', 'RITE_AVcasNet_minmax0.01_SpCL0.0085', 'RITE_AVcasNet_minmax0.01_SpCL0.09'
            # 'RITE_AVcasNet_minmax0.01_SpCL0.083', 'RITE_AVcasNet_minmax0.01_SpCL0.087'
           #'RITE_final_rrwnetspcl_0.05', 'RITE_final_rrwnetspcl_0.5', 'RITE_final_rrwnetspcl_0.01', 'RITE_final_rrwnetspcl_0.1', 'RITE_final_rrwnetspcl_1', 
           #'RITE_final_rrwnetminmax_0.19_spcl_0.05'
           #'connection_sensitive_loss'
            # 'RITE_RRWNet_BCE_base_blurred_enhanced',
            # 'RITE_RRWNet_BCE_base_gamma_bright_enhanced',
            # 'RITE_RRWNet_BCE_base_gamma_dark_enhanced',
            # 'RITE_RRWNet_our_blurred_enhanced',
            # 'RITE_RRWNet_our_gamma_bright',
            # 'RITE_RRWNet_our_gamma_dark'
            # 'RITE_RRWNet_ours_gamma_dark'
            #'flow_based_loss'
            # 'TWGANbase'
           # 'RITE_RIP-AV_results_processed'
            # 'Intra'
            # 'RITE_RIP-AV_results_processed'
            #'flow_based_loss', 'RITE_RIP-AV_results_processed', 'supervoxelloss', 'rrwnet_minmax_0.19', 'Intra'
            #'BCE_base', 'topoloss5'
            #'RITE_Superpixel_FCN_minmax_0.19_spcl_0.001', 'RITE_Superpixel_FCN_minmax_0.19_spcl_0.003', 'RITE_Superpixel_FCN_minmax_0.19_spcl_0.005', 'RITE_Superpixel_FCN_minmax_0.19_spcl_0.007',
            # 'RITE_AttUNet', 'RITE_AttUNet_minmax_0.1', "RITE_AttUNet_minmax_0.1_spcl_0.007", "RITE_CTFNet_minmax_0.5", "RITE_CTFnet_minmax_0.5_spcl_0.003", "RITE_CTFNetbase", "RITE_Iternet_minmax_1.3", "RITE_Iternet_minmax_1.3_spcl_0.0069"
           # "RITE_RollingUNetbase3", "RITE_RollingUNet_minmax_1.0", "RITE_RollingUNet_minmax_1.0_spcl_0.09", "RITE_UNet_base", "RITE_UNet_minmax_0.05", "RITE_UNet_minmax_0.05_spcl_0.0021", "RITE_Unet++_base", "RITE_UNet++_minmax_1.0", "RITE_UNet++_minmax_1.0_spcl_0.03" 
            # 'RITE_RSFConvUnet_BCE'
            # 'connection_sensitive_loss'
        })  
elif args.dataset == "LES-AV":
    dataset = constants.lesav_dataset
    dataset.paths = {
        'gt_orlando',
        'masks',
    }
    dataset.gt = 'gt_orlando'
    if args.predictions_path is None:
        dataset.paths = dataset.paths.union({
            # 'rrwnet',
            # 'base'
            # 'LESbase',
            # 'LESminmax0.1', 'LESminmax0.5', 'LESminmax1.0', 'LESminmax2.0', 
            'LES_RITE_RSFConvUnet_BCE_test'

        })
elif args.dataset == "LES-test":
    dataset = constants.lestest_dataset
    dataset.paths = {
        'gt_orlando',
        'masks',
    }
    dataset.gt = 'gt_orlando'
    if args.predictions_path is None:
        dataset.paths = dataset.paths.union({
            # 'rrwnet',
            # 'base'
            # 'LESbase',
            # 'minmax0.1', 'minmax0.5', 'minmax1.0', 'minmax2.0', 
            # 'minmax1.5'
            # 'addspCLbl3010', 'addspCLbl2010', 'addspCLbl1010',
            # 'addspCLbl0510', 'addspCLbl2510', 'addspCLbl1510'
            # 'addcl0.1', 'addcl0.5', 'addcl1.0', 
            # 'Rolling_Unetbase', 'Rolling_Unetminmax', 'Rolling_UnetSpcl',
            # 'AttUNetbase', 'AttUNetminmax', 'AttUNetSpcl', 'Iternetbase', 'Iternetminmax',  
            # 'IternetSpcl'
            # 'Unetppbase', 'Unetppminmax'
            # 'nnUNet_LES-test'
            # 'LES_AttUNet_minmax_0.1', 'LES_AttUNet_minmax_0.01', 'LES_AttUNet_minmax_0.5', 'LES_AttUNet_minmax_0.05', 'LES_AttUNet_minmax_1.0'
            # 'LES_Rolling_Unet_minmax_0.01', 'LES_Rolling_Unet_minmax_0.05', 'LES_Rolling_Unet_minmax_0.1', 'LES_Rolling_Unet_minmax_0.5', 'LES_Rolling_Unet_minmax_1.0'
            # 'LES_UNet_minmax_0.01','LES_UNet_minmax_0.05','LES_UNet_minmax_0.1','LES_UNet_minmax_0.5','LES_UNet_minmax_1.0',
            # 'LES_rrwnet_minmax_0.1_spCL_0.01','LES_rrwnet_minmax_0.1_spCL_0.03','LES_rrwnet_minmax_0.1_spCL_0.05','LES_rrwnet_minmax_0.1_spCL_0.07','LES_rrwnet_minmax_0.1_spCL_0.09',
            # 'LES_CTFNet_minmax_0.01', 'LES_CTFNet_minmax_0.1', 'LES_CTFNet_minmax_0.05', 'LES_CTFNet_minmax_0.5', 'LES_CTFNet_minmax_1.0', 
            # 'LES_AttUNet_minmax_1.0_spcl_0.01','LES_AttUNet_minmax_1.0_spcl_0.03','LES_AttUNet_minmax_1.0_spcl_0.05','LES_AttUNet_minmax_1.0_spcl_0.07','LES_AttUNet_minmax_1.0_spcl_0.09',
            # 'LES_CTFNet_minmax_0.05_spcl_0.01','LES_CTFNet_minmax_0.05_spcl_0.03','LES_CTFNet_minmax_0.05_spcl_0.05','LES_CTFNet_minmax_0.05_spcl_0.07','LES_CTFNet_minmax_0.05_spcl_0.09',
            #'LES_RollingUNet_minmax_0.05_spcl_0.01', 'LES_RollingUNet_minmax_0.05_spcl_0.03','LES_RollingUNet_minmax_0.05_spcl_0.05','LES_RollingUNet_minmax_0.05_spcl_0.07','LES_RollingUNet_minmax_0.05_spcl_0.09'
            #'LES_IterNet_minmax_1.0_spcl_0.01}', 'LES_IterNet_minmax_1.0_spcl_0.03}','LES_IterNet_minmax_1.0_spcl_0.05}','LES_IterNet_minmax_1.0_spcl_0.07}','LES_IterNet_minmax_1.0_spcl_0.09}',
            #'LES_UNet_minmax_0.01_spCL_0.01', 'LES_UNet_minmax_0.01_spCL_0.03', 'LES_UNet_minmax_0.01_spCL_0.05', 'LES_UNet_minmax_0.01_spCL_0.07', 'LES_UNet_minmax_0.01_spCL_0.09',
            # 'LES_UNet_base_1.0'
            #'LES_UNet_minmax_0.02','LES_UNet_minmax_0.03','LES_UNet_minmax_0.04','LES_UNet_minmax_0.6','LES_UNet_minmax_0.7','LES_UNet_minmax_0.8','LES_UNet_minmax_0.9',
            #'LES_RollingUNet_minmax_0.05_spcl_0.001','LES_RollingUNet_minmax_0.05_spcl_0.003','LES_RollingUNet_minmax_0.05_spcl_0.005','LES_RollingUNet_minmax_0.05_spcl_0.007','LES_RollingUNet_minmax_0.05_spcl_0.009',
            #'LES_AttUNet_minmax_1.0_spcl_0.001', 'LES_AttUNet_minmax_1.0_spcl_0.003','LES_AttUNet_minmax_1.0_spcl_0.005','LES_AttUNet_minmax_1.0_spcl_0.007','LES_AttUNet_minmax_1.0_spcl_0.009',
            #'LES_IterNet_minmax_1.0_spcl_0.001_nseg_30', 'LES_IterNet_minmax_1.0_spcl_0.003_nseg_30','LES_IterNet_minmax_1.0_spcl_0.005_nseg_30','LES_IterNet_minmax_1.0_spcl_0.007_nseg_30','LES_IterNet_minmax_1.0_spcl_0.009_nseg_30',
            #'LES_CTFNet_minmax_0.05_spcl_0.001_nseg_30','LES_CTFNet_minmax_0.05_spcl_0.003_nseg_30','LES_CTFNet_minmax_0.05_spcl_0.005_nseg_30','LES_CTFNet_minmax_0.05_spcl_0.007_nseg_30','LES_CTFNet_minmax_0.05_spcl_0.009_nseg_30', 
            # 'LES_rrwnet_minmax_0.1_spCL_0.001_nsegs_30', 'LES_rrwnet_minmax_0.1_spCL_0.003_nsegs_30','LES_rrwnet_minmax_0.1_spCL_0.005_nsegs_30','LES_rrwnet_minmax_0.1_spCL_0.007_nsegs_30',
            # 'LES_AttUNet_minmax_1.0_spcl_0.0055', 'LES_AttUNet_minmax_1.0_spcl_0.0060', 'LES_AttUNet_minmax_1.0_spcl_0.0065', 
            # 'LES_RollingUNet_minmax_0.05_spcl_0.0056', 'LES_RollingUNet_minmax_0.05_spcl_0.0057', 'LES_RollingUNet_minmax_0.05_spcl_0.0058', 'LES_RollingUNet_minmax_0.05_spcl_0.0059', 
            # 'LES_UNet_minmax_0.02_spcl_0.001', 'LES_UNet_minmax_0.02_spcl_0.003', 'LES_UNet_minmax_0.02_spcl_0.005', 'LES_UNet_minmax_0.02_spcl_0.007', 'LES_UNet_minmax_0.02_spcl_0.009', 
            # 'LES_IterNet_minmax_1.0_spcl_0.0091', 'LES_IterNet_minmax_1.0_spcl_0.0093', 'LES_IterNet_minmax_1.0_spcl_0.0095', 'LES_IterNet_minmax_1.0_spcl_0.0097', 'LES_IterNet_minmax_1.0_spcl_0.0099', 
            #'LES_rrwnet_minmax_0.1_spCL_0.02', 'LES_rrwnet_minmax_0.1_spCL_0.04', 'LES_rrwnet_minmax_0.1_spCL_0.06', 'LES_rrwnet_minmax_0.1_spCL_0.08', 
            #'LES_rrwnet_minmax_0.1_spCL_0.002', 'LES_rrwnet_minmax_0.1_spCL_0.004', 'LES_rrwnet_minmax_0.1_spCL_0.006', 'LES_rrwnet_minmax_0.1_spCL_0.008', 
            # 'LES_UNet++_minmax_1.0_spcl_0.001_nseg_30', 'LES_UNet++_minmax_1.0_spcl_0.003_nseg_30', 'LES_UNet++_minmax_1.0_spcl_0.005_nseg_30', 'LES_UNet++_minmax_1.0_spcl_0.007_nseg_30', 'LES_UNet++_minmax_1.0_spcl_0.009_nseg_30',
            # 'LES_RollingUNet_minmax_0.05_spcl_0.0051', 'LES_RollingUNet_minmax_0.05_spcl_0.0052', 'LES_RollingUNet_minmax_0.05_spcl_0.0053', 'LES_RollingUNet_minmax_0.05_spcl_0.0054', 
            # 'LES_RollingUNet_minmax_0.05_spcl_0.0061', 'LES_RollingUNet_minmax_0.05_spcl_0.0062', 'LES_RollingUNet_minmax_0.05_spcl_0.0063', 'LES_RollingUNet_minmax_0.05_spcl_0.0064', 
            # 'LES_RollingUNet_minmax_0.05_spcl_0.0067', 'LES_RollingUNet_minmax_0.05_spcl_0.0068', 'LES_RollingUNet_minmax_0.05_spcl_0.0069', 'LES_RollingUNet_minmax_0.05_spcl_0.0066', 
            # 'LES_RollingUNet_minmax_0.05_spcl_0.0071', 'LES_RollingUNet_minmax_0.05_spcl_0.0072', 'LES_RollingUNet_minmax_0.05_spcl_0.0073', 'LES_RollingUNet_minmax_0.05_spcl_0.0074', 'LES_RollingUNet_minmax_0.05_spcl_0.0075', 
            # 'LES_RollingUNet_minmax_0.05_spcl_0.002', 'LES_RollingUNet_minmax_0.05_spcl_0.004', 'LES_RollingUNet_minmax_0.05_spcl_0.006', 'LES_RollingUNet_minmax_0.05_spcl_0.008',
            # 'LES_rrwnet_spCL_1', 'LES_rrwnet_spCL_0.01', 'LES_rrwnet_spCL_0.1', 'LES_rrwnet_spCL_0.05', 'LES_rrwnet_spCL_0.5', 
            # 'minmax0.1_spCL_0.06_final'
            # 'BCEbase'
            # 'connection_sensitive_loss'
            'intra'
            # 'BCEbase', 'connection_sensitive_loss', 'flow_based_loss', 'intra','LES-test_RIP-AV_results_processed',
            # 'minmax0.1', 'topoloss5', 'TWGANbase' 
            # 'minmax0.1_spCL_0.06_final'
            # 'AttUNetbase',
            #'topoloss5'
            # 'TWGANbase' 
            #'flow_based_loss'
            #'CTFNetbase',
            # 'LES-test_RIP-AV_results_processed'
            # 'supervoxelloss'
            # 'Iternetbase','LES_AttUNet_minmax_1.0','LES_AttUNet_minmax_1.0_spcl_0.0065_nseg_30',
            # 'LES_CTFNet_minmax_0.05','LES_CTFNet_minmax_0.05_spcl_0.07','LES_Iternet_minmax_1.0','LES_IterNet_minmax_1.0_spcl_0.0093',
            # 'LES_Rolling_Unet_minmax_0.05','LES_RollingUNet_minmax_0.05_spcl_0.0078',
            # 'LES_UNet_base_1.0','LES_UNet_minmax_0.02','LES_UNet_minmax_0.02_spcl_0.007','LES_UNet++_minmax_1.0',
            # 'LES_UNet++_minmax_1.0_spcl_0.09',
            # 'Rolling_Unetbase'
            # 'LES_Iternet_minmax_1.0'
        })
elif args.dataset == "LF_private":
    dataset = constants.lf_private_dataset
    dataset.paths = {
        'gt',
        'masks',
    }
    dataset.gt = 'gt'
    if args.predictions_path is None:
        dataset.paths = dataset.paths.union({
            'LF_private_RRWNet_minmax_0.1', 'LF_private_RRWNet_minmax_0.01', 'LF_private_RRWNet_minmax_0.001','LF_private_RRWNet_minmax_0.5',
            'LF_private_RRWNet_minmax_0.05', 'LF_private_RRWNet_minmax_0.005', 'LF_private_RRWNet_minmax_1.0',
        })
elif args.dataset == "HRF":
    dataset = constants.hrf_dataset
    dataset.paths = {
        'gt_chen',
        'masks',
    }
    dataset.gt = 'gt_chen'
    if args.predictions_path is None:
        dataset.paths = dataset.paths.union({
            # 'gt2_hemelings',
            # 'rrwnet',
            # 'base',
            # 'minmax1.0', 'minmax1.1', 'minmax1.2', 'minmax1.3', 'minmax1.4', 'minmax1.5', 
            # 'minmax0.9', 'minmax0.8', 'minmax0.7', 'minmax0.6', 'minmax0.5', 'minmax0.4', 'minmax0.3', 'minmax0.2', 'minmax0.1',
            # 'addcl0.5', 'addcl0.6', 'addcl0.7', 'addcl0.8', 'addcl0.9', 'addcl1.0',
            # 'addCl0.6'
            # 'mimax1.3cl0.3', 'mimax1.3cl0.1', 'mimax1.3cl1.0', 
            # 'addspCLbl3010', 'addspCLbl2010', 'addspCLbl1010', 
            # 'addspCLbl0510', 'addspCLbl2510', 'addspCLbl1510'
            # 'addspCLbl1.0', 'addspCLbl0.01'
            # 'minmax13Spcl003', 'minmax13Spcl007', 'minmax13Spcl013', 'minmax13Spcl015', 
            # 'maskSPcl30', 'maskSPcl20', 'maskSPcl10', 
            # 'nnUNet_HRF'
           # 'supervoxelloss',
            # 'self-supervised_loss', 'bifurLoss', 'bifurwselfLoss'
            # 'maskSpcl15', 'maskSpcl05', 'maskSpcl25'
            # 'HRF_RRWNet_minmax_0.3_spcl_0.001', 'HRF_RRWNet_minmax_0.3_spcl_0.003', 'HRF_RRWNet_minmax_0.3_spcl_0.005', 'HRF_RRWNet_minmax_0.3_spcl_0.007', 'HRF_RRWNet_minmax_0.3_spcl_0.009', 
            # 'HRF_RRWNet_minmax_0.3_spcl_0.01', 'HRF_RRWNet_minmax_0.3_spcl_0.03', 'HRF_RRWNet_minmax_0.3_spcl_0.05', 'HRF_RRWNet_minmax_0.3_spcl_0.07', 'HRF_RRWNet_minmax_0.3_spcl_0.09',
            # 'HRF_AVcasNet_minmax0.01_SpCL0.005', 'HRF_AVcasNet_minmax0.01_SpCL0.007', 'HRF_AVcasNet_minmax0.01_SpCL0.009', 
            # 'HRF_RRWNet_minmax_0.3_spcl_0.02', 'HRF_RRWNet_minmax_0.3_spcl_0.04', 'HRF_RRWNet_minmax_0.3_spcl_0.06', 'HRF_RRWNet_minmax_0.3_spcl_0.08', 
            # 'HRF_RRWNet_minmax_0.3_spcl_0.002', 'HRF_RRWNet_minmax_0.3_spcl_0.004', 'HRF_RRWNet_minmax_0.3_spcl_0.006', 'HRF_RRWNet_minmax_0.3_spcl_0.008', 
            # 'HRF_RRWNet_minmax_0.3_spcl_0.0094', 'HRF_RRWNet_minmax_0.3_spcl_0.0096', 'HRF_RRWNet_minmax_0.3_spcl_0.0098', 
            # 'HRF_RRWNet_minmax_0.7_spcl_0.01', 'HRF_RRWNet_minmax_0.7_spcl_0.02', 'HRF_RRWNet_minmax_0.7_spcl_0.03', 'HRF_RRWNet_minmax_0.7_spcl_0.04', 'HRF_RRWNet_minmax_0.7_spcl_0.05', 
            # 'HRF_RRWNet_minmax_0.7_spcl_0.06', 'HRF_RRWNet_minmax_0.7_spcl_0.07', 'HRF_RRWNet_minmax_0.7_spcl_0.08', 'HRF_RRWNet_minmax_0.7_spcl_0.09', 
            # # 'HRF_RRWNet_minmax_1.0_spcl_0.01', 'HRF_RRWNet_minmax_1.0_spcl_0.02', 'HRF_RRWNet_minmax_1.0_spcl_0.03', 'HRF_RRWNet_minmax_1.0_spcl_0.04', 'HRF_RRWNet_minmax_1.0_spcl_0.05', 
            # 'HRF_RRWNet_minmax_1.0_spcl_0.06', 'HRF_RRWNet_minmax_1.0_spcl_0.07', 'HRF_RRWNet_minmax_1.0_spcl_0.08', 'HRF_RRWNet_minmax_1.0_spcl_0.09', 
            # 'HRF_RRWNet_minmax_0.7_spcl_0.005'
            # 'HRF_RRWNet_minmax0.7'
            'BCE_base'
          #'connection_sensitive_loss'
        #   'flow_based_loss'
            # 'PVN_lkw'
        #    'TWGANbase'
          # 'HRF_RIP-AV_results_1-5_preprocessed'
            # 'BCE_base', 'connection_sensitive_loss', 'flow_based_loss', 'intra','HRF_RIP-AV_results_processed',
            # 'topoloss5', 'HRF_RRWNet_minmax0.7', 'HRF_RRWNet_minmax_0.7_spcl_0.005', 'TWGANbase'
            # 'HRF_RRWNet_minmax_0.7_spcl_0.005'
            # 'HRF_RIP-AV_results_1-5_preprocessed'
            # 'AttUNetnewmm0.01'
            # 'AttUNetbase', 'AttUNetnewmm0.01', 
            # 'CTFNet_minmax_0.05', 'CTFNetbase',  
            # 'HRF_AttUNet_minmax_0.01_spcl_0.05_nseg_25', 'HRF_CTFNet_minmax_0.05_spcl_0.0032', 
            # 'HRF_Iternet_minmax_0.05', 
            # 'HRF_Rolling_Unet_spcl_0.01_nseg_25',  
            # 'HRF_UNet_minmax_0.5_spcl_0.05', 
            # 'HRF_UNet++_minmax_0.05_spcl_0.0053', 
            #'Iternet_minmax_0.05', 'Iternetbase2', 
            # 'Rolling_Unetbase', 'RollingUNet_minmax_spcl_0.01', 
            # 'UNet++minmax0.05',
            # 'UNetbase5', 'UNetnewmm0.5', 'UNetppbase'
            #'CTFNetbase'
            # "HRF_RSFConvUnet_BCE"
            })
else:
    raise ValueError

if args.predictions_path is not None:
    dataset.paths.add(args.predictions_path)
    save_fn = join(results_dir, f'{dataset.name}_model_{args.test}.json')
else:
    save_fn = join(results_dir, f'{dataset.name}_{args.test}_{args.version}.json')
assert Path(save_fn).parent.exists(), Path(save_fn).parent

# add samples to dataset Namespace
dataset.samples = {}

for path in dataset.paths:
    full_path = data_path / dataset.name / path
    print(full_path)
    assert full_path.exists(), full_path
    for id_ in dataset.ids:
        sample = {
            path: str(full_path / dataset.pattern.format(id_=id_))
        }
        if id_ not in dataset.samples:
            dataset.samples[id_] = sample
        else:
            dataset.samples[id_].update(sample)


paths = dataset.paths
paths = { p for p in paths if 'masks' not in p and 'gt' not in p }

if args.pixels == 'both':
    results = {
        'All': {},
        'Intersection': {},
    }
elif args.pixels == 'all':
    results = {
        'All': {},
    }
elif args.pixels == 'intersection':
    results = {
        'Intersection': {},
    }
else:
    raise ValueError("Invalid value for --pixels")

if args.test == 'mav':
    test_func = test_utils.mav
elif args.test == 'save_all':
    test_func = test_utils.save_all
elif args.test == 'topo_mp':
    test_func = test_utils.topo_mp
else:
    raise ValueError


if 'All' in results.keys():
    for i, path in enumerate(paths):
        results['All'][path] = test_func(
            dataset,
            path,
            results_dir,
            gt_key=dataset.gt,
            n_paths=args.n_paths,
        )
if 'Intersection' in results.keys():
    for i, path in enumerate(paths):
        results['Intersection'][path] = test_func(
            dataset,
            path,
            results_dir,
            predicted_only=True,
            gt_key=dataset.gt,
            n_paths=args.n_paths,
        )

with open(save_fn, 'w') as f:
    json.dump(results, f, indent=4)
