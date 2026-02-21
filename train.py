from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
from models import Backbone as Backbone
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
from torchvision.ops import roi_align
from glob import glob
import torchvision
import os, random, cv2, argparse
from hparams import hparams, get_image_list
from models import model as flowmodel

parser = argparse.ArgumentParser(description='Code to train the model')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True, type=str)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--syncnet_checkpoint_path', help='Load the pre-trained Expert discriminator', required=True, type=str)

parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default=None, type=str)

args = parser.parse_args()


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16


class perceptionLoss():
    def __init__(self, device):
        vgg = torchvision.models.vgg19(pretrained=True)
        vgg.eval()
        self.features = vgg.features.to(device)
        self.feature_layers = ['4', '9', '18', '27', '36']
        self.mse_loss = nn.MSELoss()

    def getfeatures(self, x):
        feature_list = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.feature_layers:
                feature_list.append(x)
        return feature_list

    def calculatePerceptionLoss(self, video_pd, video_gt):
        features_pd = self.getfeatures(video_pd.view(video_pd.size(0)*video_pd.size(2), video_pd.size(1), video_pd.size(3), video_pd.size(4)))
        features_gt = self.getfeatures(video_gt.view(video_gt.size(0)*video_gt.size(2), video_gt.size(1), video_gt.size(3), video_gt.size(4)))
        
        with torch.no_grad():
            features_gt = [x.detach() for x in features_gt]
        
        perceptual_loss = sum([self.mse_loss(features_pd[i], features_gt[i]) for i in range(len(features_gt))])
        return perceptual_loss

class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)
        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.lms'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def get_lms_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        landmark_path = dirname(start_frame)
        # landmark_path = vidname.replace('lrs2_preprocessed','landmark_file')
        windows_landmark = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame_lms = join(landmark_path,'{}.lms'.format(frame_id))
            if not isfile(frame_lms):
                return None
            windows_landmark.append(frame_lms)
        return windows_landmark
    
    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            fname = fname.replace('lms','jpg')
            fname = fname.replace('landmark_file','lrs2_preprocessed')
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def get_landmark_roi(self, window_landmark,window_fnames):
        rois_mouths = []

        for idx,(fname_lms,fname) in enumerate(zip(window_landmark,window_fnames)):
            fname = fname.replace('lms','jpg')
            fname = fname.replace('landmark_file','lrs2_preprocessed')
            img = cv2.imread(fname)
            x_scale,y_scale = 96/img.shape[1],96/img.shape[0]

            landmarks = np.loadtxt(fname_lms).astype(np.int64)
            landmarks_mouth = landmarks[48:]
            x1, y1, w, h = cv2.boundingRect(landmarks_mouth)
            x2,y2 = x1+w,y1+h
            roi = torch.tensor([x1*x_scale, y1*y_scale, x2*x_scale, y2*y_scale]).unsqueeze(0)
            img_inds = roi.new_full((1,1),idx)
            rois = torch.cat([img_inds,roi],dim=-1)
            rois_mouths.append(rois)
        rois_mouths = torch.cat(rois_mouths, 0)
        return rois_mouths

    def get_landmark(self, window_landmark,window_fnames):
        windows = []
        mask_windows = []
        for fname_lms,fname in zip(window_landmark,window_fnames):
            fname = fname.replace('lms','jpg')
            fname = fname.replace('landmark_file','lrs2_preprocessed')
            img = cv2.imread(fname)
            landmarks = np.loadtxt(fname_lms).astype(np.int64)
            landmarks_mouth = landmarks[48:]
            x1, y1, w, h = cv2.boundingRect(landmarks_mouth)
            x2,y2 = x1+w,y1+h
            landmarks_mouth = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])

            images_zero = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
            mask = cv2.fillConvexPoly(images_zero, landmarks_mouth, (255,255,255))
            img2_fg = cv2.bitwise_and(img,img,mask = mask)
            img2_fg = cv2.resize(img2_fg, (hparams.img_size, hparams.img_size))
            mask =cv2.resize(mask, (hparams.img_size, hparams.img_size))
            mask_windows.append(mask)
            windows.append(img2_fg)
        return windows,mask_windows
    
    def prepare_window(self, window):
        # 3 x T x H x W
        # print(np.asarray(window).shape)
        
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.lms')))
            if len(img_names) <= 3 * syncnet_T:
                continue
            
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            window_landmark = self.get_window(img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue
            gt_roi = self.get_landmark_roi(window_landmark,window_fnames)
            gt_landmark,mask_window = self.get_landmark(window_landmark,window_fnames)
            window = self.read_window(window_fnames)
            
            if window is None:
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                continue

            try:
                
                # wavpath = join(os.path.dirname(vidname).replace('landmark_file','hdtf_preprocessed/new_format'), "audio.wav")
                wavpath = join(vidname.replace('landmark_file','lrs2_preprocessed'), "audio.wav")

                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)
            
            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None: continue
            lms = self.prepare_window(gt_landmark)
            window = self.prepare_window(window)
            y = window.copy()
            window[:, :, window.shape[2]//2:] = 0.

            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(y)
            lms = torch.FloatTensor(lms)
            mask_window = np.array(mask_window)/255
            mask = torch.FloatTensor(mask_window)
            return x, indiv_mels, mel, y, gt_roi, mask, lms

def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

device = torch.device("cuda" if use_cuda else "cpu")
syncnet = SyncNet().to(device)
for p in syncnet.parameters():
    p.requires_grad = False

recon_loss = nn.L1Loss()
MSE_loss = nn.MSELoss()
disc_emo = perceptionLoss(device)
# flow_model = flowmodel.UNet(6,4)
# flow_model.to(device)
# for p in flow_model.parameters():
#     p.requires_grad = False
    
def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)

def train(device, model, flow_model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    resumed_step = global_step
 
    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_sync_loss, running_l1_loss = 0., 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, indiv_mels, mel, gt, rois, mask, lms) in prog_bar:
            model.train()
            optimizer.zero_grad()

            # Move data to CUDA device
            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)
            lms = lms.to(device)
            mask =  mask.to(device)
            rois = rois.to(device)
            g = model(indiv_mels, x)


            batch_size, channels, T, height, width = g.shape
            total_loss_flow_0_1 = 0
            total_loss_flow_1_0 = 0

            # 遍历序列，计算相邻帧之间的光流
            for i in range(T - 1):
                # 提取相邻的两帧：第 i 帧和第 i+1 帧
                frame_i = g[:, :, i, :, :]
                frame_next = g[:, :, i+1, :, :]
                
                gt_frame_i = gt[:, :, i, :, :].detach()
                gt_frame_next = gt[:, :, i+1, :, :].detach()

                # 计算光流
                g_flow = flow_model(torch.cat((frame_i, frame_next), dim=1))
                gt_flow = flow_model(torch.cat((gt_frame_i, gt_frame_next), dim=1))

                current_mask = mask[:, i, :, :].unsqueeze(1).repeat(1, 2, 1, 1) 

                g_flow_0_1, g_flow_1_0 = g_flow[:, :2, :, :] * current_mask, g_flow[:, 2:, :, :] * current_mask
                gt_flow_0_1, gt_flow_1_0 = gt_flow[:, :2, :, :] * current_mask, gt_flow[:, 2:, :, :] * current_mask

                # 累加损失
                total_loss_flow_0_1 += recon_loss(g_flow_0_1, gt_flow_0_1)
                total_loss_flow_1_0 += recon_loss(g_flow_1_0, gt_flow_1_0)

            # 最后取平均值作为最终 loss
            loss_flow_0_1 = total_loss_flow_0_1 / (T - 1)
            loss_flow_1_0 = total_loss_flow_1_0 / (T - 1)

            # g_flow = flow_model(torch.cat((g[:,:,0,:,:],g[:,:,-1,:,:]), dim= 1))
            # gt_flow = flow_model(torch.cat((gt[:,:,0,:,:].detach(),gt[:,:,-1,:,:].detach()), dim= 1))
            # mask_flow = mask[:,2,:,:].unsqueeze(1).repeat(1,2,1,1)
            # g_flow_0_1, g_flow_1_0 = g_flow[:,:2,:,:] * mask_flow, g_flow[:,2:,:,:] * mask_flow
            # gt_flow_0_1, gt_flow_1_0 = gt_flow[:,:2,:,:] * mask_flow, gt_flow[:,2:,:,:] * mask_flow
            
            # loss_flow_0_1, loss_flow_1_0 = recon_loss(g_flow_0_1, gt_flow_0_1), recon_loss(g_flow_1_0, gt_flow_1_0)
            
            if hparams.syncnet_wt > 0.:
                sync_loss = get_sync_loss(mel, g)
            else:
                sync_loss = 0.

            l1loss = recon_loss(g, gt)
            
            # mouth_loss = 0
            # for bs in range(g.shape[0]):
                
            #     fake_roi = roi_align(input=g[bs].permute(1, 0, 2, 3), boxes=rois[bs],output_size=40)
            #     real_ori = roi_align(input=gt[bs].permute(1, 0, 2, 3), boxes=rois[bs],output_size=40)
            #     mouth_loss += recon_loss(fake_roi,real_ori)
            ploss =  disc_emo.calculatePerceptionLoss(g,gt)
            # mouth_loss = mouth_loss/g.shape[0]
            # print(g.shape,gt.shape) ##(16,3,5,96,96)
            mask = mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
            mouth_loss = recon_loss(mask*g,lms) 
            # mouth_loss = recon_loss(g,lms)

            loss = hparams.syncnet_wt * sync_loss + (1 - hparams.syncnet_wt - hparams.mouth_wt - hparams.pl_wt - hparams.tem_wt*2) * l1loss 
            +  hparams.mouth_wt * mouth_loss + hparams.pl_wt * ploss + hparams.tem_wt * loss_flow_0_1 + hparams.tem_wt * loss_flow_1_0
            # loss = hparams.syncnet_wt * sync_loss + (1 - hparams.syncnet_wt - hparams.mouth_wt - hparams.pl_wt) * l1loss 
            # + hparams.mouth_wt * mouth_loss + hparams.pl_wt * ploss
            
            # loss = hparams.syncnet_wt * sync_loss + (1 - hparams.syncnet_wt - hparams.mouth_wt -  hparams.tem_wt*2) * l1loss 
            # +  hparams.mouth_wt * mouth_loss + hparams.tem_wt * loss_flow_0_1 + hparams.tem_wt * loss_flow_1_0

            # loss = hparams.syncnet_wt * sync_loss + (1 - hparams.syncnet_wt - hparams.pl_wt - hparams.tem_wt*2) * l1loss 
            # + hparams.pl_wt * ploss + hparams.tem_wt * loss_flow_0_1 + hparams.tem_wt * loss_flow_1_0
            loss.backward()
            optimizer.step()

            if global_step % checkpoint_interval == 0:
                save_sample_images(x, g, gt, global_step, checkpoint_dir)

            global_step += 1
            cur_session_steps = global_step - resumed_step

            running_l1_loss += l1loss.item()
            if hparams.syncnet_wt > 0.:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, flow_model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step == 1 or global_step % hparams.eval_interval == 0:
                with torch.no_grad():
                    average_sync_loss = eval_model(test_data_loader, global_step, device, model, checkpoint_dir)

                    if average_sync_loss < .75:
                        hparams.set_hparam('syncnet_wt', 0.01) # without image GAN a lesser weight is sufficient

            prog_bar.set_description('L1: {}, Sync Loss: {}'.format(running_l1_loss / (step + 1),
                                                                    running_sync_loss / (step + 1)))

        global_epoch += 1
        

def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 700
    print('Evaluating for {} steps'.format(eval_steps))
    sync_losses, recon_losses = [], []
    step = 0
    while 1:
        for x, indiv_mels, mel, gt,_,_,_ in test_data_loader:
            step += 1
            model.eval()

            # Move data to CUDA device
            x = x.to(device)
            gt = gt.to(device)
            indiv_mels = indiv_mels.to(device)
            mel = mel.to(device)

            g = model(indiv_mels, x)

            sync_loss = get_sync_loss(mel, g)
            l1loss = recon_loss(g, gt)

            sync_losses.append(sync_loss.item())
            recon_losses.append(l1loss.item())

            if step > eval_steps: 
                averaged_sync_loss = sum(sync_losses) / len(sync_losses)
                averaged_recon_loss = sum(recon_losses) / len(recon_losses)

                print('L1: {}, Sync loss: {}'.format(averaged_recon_loss, averaged_sync_loss))

                return averaged_sync_loss

def save_checkpoint(model, flow_model, optimizer, step, checkpoint_dir, epoch):

    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

    torch.save({
        "state_dict": flow_model.state_dict(),
    }, join(checkpoint_dir,'flow.pth'))

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model

if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir

    # Dataset and Dataloader setup
    train_dataset = Dataset('train')
    test_dataset = Dataset('val')

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.batch_size,
        num_workers=4)

    device = torch.device("cuda" if use_cuda else "cpu")
    flow_model = flowmodel.UNet(6,4)
    flow_model.to(device)
    # Model
    model = Backbone().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    trainable_params = []
    trainable_params.extend(model.parameters())
    trainable_params.extend(flow_model.parameters())
    optimizer = optim.Adam([p for p in trainable_params if p.requires_grad],
                           lr=hparams.initial_learning_rate)

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)
        
    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, overwrite_global_states=False)
    dict1 = torch.load('SuperSloMo.ckpt')
    flow_model.load_state_dict(dict1['state_dictFC'])
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # Train!
    train(device, model, flow_model, train_data_loader, test_data_loader, optimizer,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs)
