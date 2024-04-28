import os
from matplotlib import pyplot as plt
import torch
import tqdm
from core.base_model import BaseModel
from core.logger import LogTracker
import copy
from torch.optim.lr_scheduler import LinearLR

class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Palette(BaseModel):
    def __init__(
        self, 
        networks, 
        losses, 
        sample_num, 
        task, 
        optimizers,
        lr_schedulers=None,
        ema_scheduler=None,
        cond_on_mask=False,
        **kwargs
    ):
        ''' must to init BaseModel with kwargs '''
        super(Palette, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = losses[0]
        self.netG = networks[0]
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None
        
        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
        self.load_networks()

        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        self.optimizers.append(self.optG)
        if lr_schedulers is not None:
            self.schedulers.append(LinearLR(self.optG, **lr_schedulers[0]))

        self.resume_training() 

        if self.opt['distributed']:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase)

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task

        self.cond_on_mask = cond_on_mask
        
    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.cond_image = self.set_device(data.get('cond_image'))
        self.gt_image = self.set_device(data.get('gt_image'))
        self.gt_normal_image = self.set_device(data.get('gt_normal_image'))
        self.gt_height_image = self.set_device(data.get('gt_height_image'))
        self.footprint = self.set_device(data.get('footprint'))
        self.mask = self.set_device(data.get('mask'))
        self.mask_image = data.get('mask_image')
        self.path = data['path']
        self.batch_size = len(data['path'])
        self.height_range = data.get('height_range')
        self.mid_height = data.get('mid_height')
        
    def get_current_visuals(self, phase='train'):
        dict = {}
        if self.task in ['inpainting','uncropping'] and self.mask is not None and self.mask_image is not None:
            dict.update({
                'mask': self.mask.detach()[:].float().cpu(),
                'mask_image': (self.mask_image+1)/2,
            })
        if phase != 'train':
            dict.update({
                'output': (self.output.detach()[:].float().cpu()+1)/2
            })
        dict.update({
            'gt_image': (self.gt_image.detach()[:].float().cpu()+1)/2,
            'cond_image': (self.cond_image.detach()[:].float().cpu()+1)/2,
        })
        return dict

    def convert_to_colormap(self, image, cmap):
        image = image.squeeze(0).numpy()
        cmapped = plt.get_cmap(cmap)(image)

        # The returned array from the colormap has shape (H, W, 4) (RGBA image).
        # We convert it back to PyTorch tensor and get rid of the alpha channel 
        # assuming you want a (3, H, W) tensor.
        return torch.from_numpy(cmapped[:, :, :3]).permute(2, 0, 1)

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            mid_height, height_range = self.mid_height[idx], self.height_range[idx]

            ret_path.append('GT_{}'.format(self.path[idx]))
            gt_image = torch.clamp(self.gt_image[idx], -1, 1)
            gt_img = gt_image.detach().float().cpu()
            gt_img = torch.clamp(gt_img, -1, 1)
            gt_mask = gt_img > -1
            gt_img[gt_mask] = gt_img[gt_mask] * 0.5 * height_range + mid_height
            gt_img[~gt_mask] = 0
            ret_result.append(gt_img)

            ret_path.append('Cond_{}'.format(self.path[idx]))
            cond_img = torch.clamp(self.cond_image[idx], -1, 1)
            cond_img = cond_img.detach().float().cpu()
            cond_mask = cond_img > -1
            cond_img[cond_mask] = cond_img[cond_mask] * 0.5 * height_range + mid_height
            cond_img[~cond_mask] = 0
            ret_result.append(cond_img)

            ret_path.append(self.path[idx])
            output = torch.clamp(self.output[idx], -1, 1)
            output = output.detach().float().cpu()
            out_mask = output > -0.95
            output[out_mask] = output[out_mask] * 0.5 * height_range + mid_height
            output[~out_mask] = 0
            output[output < 0] = 0
            ret_result.append(output)

            if self.sample_num > 0:
                for k in range(self.sample_num + 1):
                    ret_path.append('Inter_{}_{}'.format(k, self.path[idx]))
                    ret_result.append(self.visuals[k * self.batch_size + idx].detach().float().cpu())

        
        if self.task in ['inpainting','uncropping'] and self.mask is not None:
            ret_path.extend(['Mask_{}'.format(name) for name in self.path])
            ret_result.extend(self.mask.detach().float().cpu())

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
        for train_data in tqdm.tqdm(self.phase_loader):
            self.set_input(train_data)
            self.optG.zero_grad()
            if self.cond_on_mask:
                mask_channel = self.mask.clone()
                mask_channel[mask_channel == 0] = -1
                cond_image = torch.cat((self.cond_image, mask_channel), dim=1)
            else:
                cond_image = self.cond_image
            loss = self.netG(self.gt_image, cond_image, mask=self.mask)
            loss.backward()
            self.optG.step()

            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals().items():
                    self.writer.add_images(key, value)
                # Log learning rate
                self.writer.add_scalar('Learning Rate', self.optG.param_groups[0]['lr'])
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)
            for scheduler in self.schedulers:
                scheduler.step()
        return self.train_metrics.result()
    
    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)

                if self.cond_on_mask:
                    mask_channel = self.mask.clone()
                    mask_channel[mask_channel == 0] = -1
                    cond_image = torch.cat((self.cond_image, mask_channel), dim=1)
                else:
                    cond_image = self.cond_image

                if self.opt['distributed']:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.module.restoration(cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.module.restoration(cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.restoration(cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.restoration(cond_image, sample_num=self.sample_num)
                    
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')

                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.val_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='val').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())

        return self.val_metrics.result()

    def test(self):
        self.netG.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for phase_data in tqdm.tqdm(self.phase_loader):
                self.set_input(phase_data)

                if self.cond_on_mask:
                    mask_channel = self.mask.clone()
                    mask_channel[mask_channel == 0] = -1
                    cond_image = torch.cat((self.cond_image, mask_channel), dim=1)
                else:
                    cond_image = self.cond_image

                if self.opt['distributed']:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.module.restoration(cond_image, y_t=None,
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.module.restoration(cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.restoration(cond_image, y_t=None, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.restoration(cond_image, sample_num=self.sample_num)
                        
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='test')
                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.test_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='test').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())
        
        test_log = self.test_metrics.result()
        ''' save logged informations into log dict ''' 
        test_log.update({'epoch': self.epoch, 'iters': self.iter})

        ''' print logged informations to the screen and tensorboard ''' 
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))

    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)

    def save_everything(self):
        """ load pretrained model and training state. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        self.save_training_state()

