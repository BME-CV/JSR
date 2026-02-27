import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets
import numpy as np
import torch
import nibabel as nib
import pandas as pd
from model.reg import Reg_Decoder,multi_scale_Reg_Decoder,multi_scale_Reg_Decoder_nearest
from model.share import Encoder
import time
import argparse
from tqdm import tqdm
from model.seg import SegDecoder

from evaluation import evaluation as mt

def strip_module_key(state_dict):
    """remove 'module.' prefix if present (DataParallel)"""
    new_state = {}
    for k, v in state_dict.items():
        nk = k
        if k.startswith("module."):
            nk = k[len("module."):]
        new_state[nk] = v
    return new_state

def dice_torch(pred, target, eps=1e-6):
    """pred and target are binary torch tensors with same shape (B, D, H, W) or (D,H,W)"""
    pred = pred.float()
    target = target.float()
    inter = (pred * target).sum(dim=[-3, -2, -1])
    union = pred.sum(dim=[-3, -2, -1]) + target.sum(dim=[-3, -2, -1])
    dice = (2.0 * inter + eps) / (union + eps)
    return dice  # per-sample Dice (tensor)

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    # try multiple key names for compatibility
    enc_state = ckpt.get("encoder", ckpt.get("endc_state_dict", None))
    reg_state = ckpt.get("reg_head", ckpt.get("reg_state_dict", None))
    return enc_state, reg_state, ckpt

def main():
    parser = argparse.ArgumentParser(description='Test model for Seg-Reg')
    parser.add_argument('--base_filters', type=int, default=8,
                        help='Base number of filters')
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of segmentation classes (must be 2 for binary segmentation)')
    # parser.add_argument("--checkpoint", required=True, help="path to saved model (.pt)")
    args = parser.parse_args()

    test_dir = '/mnt/sda/download/amos22/amos22/Test_CT'
    atlas = '/mnt/sda/download/amos22/amos22/Val_CT/amos_0223.nii.gz'


    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # weights = [10,1]
    # model_folder = ''   #.format(weights[0], weights[1])
    #注意dataset中测试数据是新或者旧，以及注意修改对应的excel命名
    # modelname='0924Joint192/Joint_best_epoch23.pth'
    # modelname='1022_multi_Joint_1_1_0.1/Joint_best_epoch59.pth'
    modelname='1207_Joint_1-1-1/Joint_best_epoch198.pth'
    model_dir = 'checkpoints/' + modelname
    savepth = 'results/' + modelname
    organ = 'iter'
    if not os.path.exists(savepth):
        os.makedirs(savepth)
    img_size = (64, 192, 192)

    Endc = Encoder(in_channel=1, first_out_channel=args.base_filters)
    Seg = SegDecoder(num_classes=args.num_classes, base_filters=args.base_filters).to(device)
    # Reg = Reg_Decoder(inshape=img_size, channels=args.base_filters * 2)

    # Reg = multi_scale_Reg_Decoder(inshape=img_size, channels=args.base_filters * 2)
    Reg = multi_scale_Reg_Decoder_nearest(inshape=img_size, channels=args.base_filters * 2)

    Endc.to(device)
    Reg.to(device)

    ckpt = torch.load(model_dir, map_location=device)
    Endc.load_state_dict(ckpt['encoder'])
    Seg.load_state_dict(ckpt['seg_head'])
    Reg.load_state_dict(ckpt['reg_head'])


    Endc.eval()
    Reg.eval()
    Seg.eval()
    save = False

    columns = ['Sample Name', 'Deformed DSC', 'Raw DSC', 'HD95','ASSD','Jacobian Det < 0']
    results_df = pd.DataFrame(columns=columns)

    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.to(device)
    # spat = utils.SpatialTransformer(size=img_size).to(device)
    #
    # test_composed = transforms.Compose([
    #     trans.NumpyType((np.float32, np.float32)),
    #                                     ])
    test_set = datasets.RegValDataset(test_dir, atlas, transforms=None)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)


    eval_hd_def = utils.AverageMeter()
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_ASSD_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()

    times=[]
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch in pbar:
            x, y, x_seg, y_seg,x_name= batch
            x = x.to(device)
            y = y.to(device)
            x_seg = x_seg.to(device)
            y_seg = y_seg.to(device)

            torch.cuda.synchronize()  # 同步设备
            start_time = time.time()

            feat_x = Endc(x)
            feat_y = Endc(y)
            final_mov, seg_mov = Seg(feat_x)
            final_fix, seg_fix = Seg(feat_y)

            seg_mov[0] = final_mov
            seg_fix[0] = final_fix

            output= Reg(x,final_mov,final_fix, feat_x, feat_y)
            # output= Reg(x, feat_x, feat_y)


            torch.cuda.synchronize()  # 同步设备
            end_time = time.time()
            times.append((end_time - start_time) * 1000)

            #--------new-----------------
            # warp_x = output[0].squeeze().squeeze().cpu().numpy()
            # x_feats = torch.mean(output[2][0], dim=1).squeeze().squeeze().cpu().numpy()
            # y_feats = torch.mean(output[3][0], dim=1).squeeze().squeeze().cpu().numpy()
            # x_f_tran = np.transpose(x_feats , (1, 2, 0))
            # y_f_tran = np.transpose(y_feats , (1, 2, 0))
            if save == True:
                # output = model(x,y)
                # flow_list = output[4]
                # --------new-----------------
                warp_x = output[0].squeeze().squeeze().cpu().numpy()
                field = output[1].squeeze().cpu().numpy()
                # fl4 = output[4][0].squeeze().cpu().numpy()
                # fl3 = output[4][1].squeeze().cpu().numpy()
                # fl2 = output[4][2].squeeze().cpu().numpy()
                warp_x = np.transpose(warp_x, (1, 2, 0))
                field = np.transpose(field, (2, 3, 1, 0))
                # fl4 = np.transpose(fl4, (2, 3, 1, 0))
                # fl3 = np.transpose(fl3, (2, 3, 1, 0))
                # fl2 = np.transpose(fl2, (2, 3, 1, 0))

                lpa1 = nib.load(os.path.join(test_dir + '/affine223CT', x_name[0]))
                affine1 = lpa1.affine
                header1 = lpa1.header
                img = nib.Nifti1Image(warp_x, affine1, header=header1)
                field = nib.Nifti1Image(field, affine1, header=header1)
                # fl4 = nib.Nifti1Image(fl4, affine1, header=header1)
                # fl3 = nib.Nifti1Image(fl3, affine1, header=header1)
                # fl2= nib.Nifti1Image(fl2, affine1, header=header1)

                nib.save(img, os.path.join(savepth, x_name[0].replace('.nii', '2')+'223.nii.gz' ))
                nib.save(field, os.path.join(savepth, 'field' + x_name[0].replace('.nii', '2') +'223.nii.gz'))
                # nib.save(fl4, os.path.join(savepth, 'ffl4' + x_name[0].replace('.nii', '2') + y_name[0]))
                # nib.save(fl3, os.path.join(savepth, 'ffl3' + x_name[0].replace('.nii', '2') + y_name[0]))
                # nib.save(fl2, os.path.join(savepth, 'ffl2' + x_name[0].replace('.nii', '2') + y_name[0]))
                #-----------以下是变形子场保存代码-------------------------------
                # for i in range(len(output[5][0])):
                #     ffl = output[5][0][i].squeeze().cpu().numpy()
                #     ffl = np.transpose(ffl, (2, 3, 1, 0))
                #     ffl = nib.Nifti1Image(ffl, affine1, header=header1)
                #     nib.save(ffl, os.path.join(savepth, f'ffl4{i}' + x_name[0].replace('.nii', '2') + y_name[0]))
                # for i in range(len(output[5][1])):
                #     ffl = output[5][1][i].squeeze().cpu().numpy()
                #     ffl = np.transpose(ffl, (2, 3, 1, 0))
                #     ffl = nib.Nifti1Image(ffl, affine1, header=header1)
                #     nib.save(ffl, os.path.join(savepth, f'ffl3{i}' + x_name[0].replace('.nii', '2') + y_name[0]))
                # for i in range(len(output[5][2])):
                #     ffl = output[5][2][i].squeeze().cpu().numpy()
                #     ffl = np.transpose(ffl, (2, 3, 1, 0))
                #     ffl = nib.Nifti1Image(ffl, affine1, header=header1)
                #     nib.save(ffl, os.path.join(savepth, f'ffl2{i}' + x_name[0].replace('.nii', '2') + y_name[0]))
                # for i in range(len(output[5][3])):
                #     ffl = output[5][3][i].squeeze().cpu().numpy()
                #     ffl = np.transpose(ffl, (2, 3, 1, 0))
                #     ffl = nib.Nifti1Image(ffl, affine1, header=header1)
                #     nib.save(ffl, os.path.join(savepth, f'ffl1{i}' + x_name[0].replace('.nii', '2') + y_name[0]))


            #评价指标
            def_out = reg_model([x_seg.float().to(device), output[1].to(device)])
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(output[1].detach().cpu().numpy()[0, :, :, :, :])
            jac_det_neg_ratio = np.sum(jac_det <= 0) / np.prod(tar.shape)
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            # if save==True:
            #     warp_seg_x = def_out.squeeze().squeeze().cpu().numpy()
            #     warp_seg_x = np.transpose(warp_seg_x, (1, 2, 0))
            #     warp_seg_x = nib.Nifti1Image(warp_seg_x, affine1, header=header1)
            #     nib.save(warp_seg_x, os.path.join(savepth, 'Wseg' + x_name[0].replace('.nii', '2') + y_name[0]))
            if organ =='liver':
                a = 4
            elif organ =='spleen':
                a = 1
            elif organ =='left':
                a = 2
            elif organ =='right':
                a = 3
            dsc = mt.compute_dice(def_out.long(),y_seg.long())[0]
            # print(dsc)
            dsc_raw = mt.compute_dice(x_seg.long(),y_seg.long())[0]
            # print(dsc_raw)
            # print(x_seg.shape)
            hd95 = mt.compute_hd95(def_out.long().cpu().numpy().squeeze(),y_seg.long().cpu().numpy().squeeze())[0]
            # print(hd95)
            ASSD = mt.compute_ASSD(def_out.long().cpu().numpy().squeeze(),y_seg.long().cpu().numpy().squeeze())[0]
            # break
            print('Deform dsc: {:.4f}, Raw dsc: {:.4f}, hd95: {:.4f},ASSD: {:.4f}'.format(dsc,dsc_raw,hd95,ASSD))
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(tar.shape)))
            eval_dsc_def.update(dsc.item() if isinstance(dsc, torch.Tensor) else dsc , x.size(0))
            eval_dsc_raw.update(dsc_raw.item() if isinstance(dsc_raw, torch.Tensor) else dsc_raw , x.size(0))
            eval_hd_def.update(hd95, x.size(0))
            eval_ASSD_raw.update(ASSD, x.size(0))
            sample_name = x_name[0]
            results_df = pd.concat([results_df, pd.DataFrame([{
                'Sample Name': sample_name,
                'Deformed DSC': dsc.item() if isinstance(dsc, torch.Tensor) else dsc,
                'Raw DSC': dsc_raw.item() if isinstance(dsc_raw, torch.Tensor) else dsc_raw,
                'HD95': hd95.item() if isinstance(hd95, torch.Tensor) else hd95,
                'ASSD': ASSD.item() if isinstance(ASSD, torch.Tensor) else ASSD,
                'Jacobian Det < 0': jac_det_neg_ratio.item() if isinstance(jac_det_neg_ratio, torch.Tensor) else jac_det_neg_ratio
            }])], ignore_index=True)

        average_time = sum(times) / len(times)
        print(f"使用实际样本的平均推理时间: {average_time:.2f} ms")

        save_path = f"{savepth}{organ}_results.xlsx"
        results_df.to_excel(save_path, index=False)

        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}，HD95: {:.3f} +- {:.3f},ASSD: {:.3f} +- {:.3f}'.
              format(eval_dsc_def.avg,
                            eval_dsc_def.std,
                            eval_dsc_raw.avg,
                            eval_dsc_raw.std,
                     eval_hd_def.avg,
                     eval_hd_def.std,
                    eval_ASSD_raw.avg,
                    eval_ASSD_raw.std
                     ))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    # GPU_iden = 1
    # GPU_num = torch.cuda.device_count()
    # print('Number of GPU: ' + str(GPU_num))
    # for GPU_idx in range(GPU_num):
    #     GPU_name = torch.cuda.get_device_name(GPU_idx)
    #     print('GPU #' + str(GPU_idx) + ': ' + GPU_name)
    # torch.cuda.set_device(GPU_iden)
    # GPU_avai = torch.cuda.is_available()
    # print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    # print('If the GPU is available? ' + str(GPU_avai))
    main()