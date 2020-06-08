import time
import os
import logging
import paddle
import paddle.fluid as fluid
import numpy as np
import cv2
from .eval import eval_metric


class Runnner(object):
    """
    A training helper for Pytorch.
    """

    def __init__(self,
                 model,
                 dataset,
                 batch_size,
                 val_dataset=None,
                 val_batch_size=50,
                 optimizer_config=None,
                 checkpoint_config=None,
                 multi_gpus=False):
        self.model = model
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        self.logger = self.init_logger()
        self.eval_type = checkpoint_config['eval_type']
        self.multi_gpus = multi_gpus

    def _add_file_handler(self,
                          logger,
                          filename=None,
                          mode='w',
                          level=logging.INFO):
        file_handler = logging.FileHandler(filename, mode)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        return logger

    def init_logger(self, level=logging.INFO):
        """Init the logger.
        Returns:
            :obj:`~logging.Logger`: Python logger.
        """
        self.log_interval = self.checkpoint_config['log_interval']
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s', level=level)
        logger = logging.getLogger(__name__)
        filename = '{}.log'.format(time.strftime('%Y%m%d_%H%M%S', time.localtime()))
        if not os.path.exists(self.checkpoint_config['work_dir']):
            os.makedirs(self.checkpoint_config['work_dir'])
        log_file = os.path.join(self.checkpoint_config['work_dir'], filename)
        self._add_file_handler(logger, log_file, level=level)
        return logger

    def init_optimizer(self):
        iter_per_epoch = int(self.dataset.length()/self.batch_size)
        bds = [iter_per_epoch * epoch for epoch in self.optimizer_config['decay_epoch']]
        base_lr = self.optimizer_config['lr']
        lrs = [base_lr * (self.optimizer_config['decay']**i) for i in range(len(bds) + 1)]
        
        lr = fluid.dygraph.PiecewiseDecay(
            boundaries=bds,
            values=lrs,
            begin=0)
        lr = fluid.layers.linear_lr_warmup(
            learning_rate=lr,
            warmup_steps=self.optimizer_config['warmup_iter'],
            start_lr=0.0,
            end_lr=base_lr)
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr, parameter_list=self.model.parameters(),
            regularization=fluid.regularizer.L2Decay(regularization_coeff=self.optimizer_config['regularization']))
        return optimizer

    def init_model(self):
        if self.checkpoint_config['load_from'] is not None:
            self.logger.info('Load checkpoint from: {}'.format(self.checkpoint_config['load_from']))
            checkpoint, _ = fluid.dygraph.load_dygraph(self.checkpoint_config['load_from'])
            self.model.set_dict(checkpoint)
        else:
            pass

    def save_checkpoint(self, epoch, iter, is_best=False):
        if is_best:
            checkpoint_file = 'Best_model'
        else:
            checkpoint_file = 'epoch_{}_iter_{}'.format(epoch, iter)
        checkpoint_dir = os.path.join(self.checkpoint_config['work_dir'], checkpoint_file)
        fluid.dygraph.save_dygraph(self.model.state_dict(), checkpoint_dir)

    def _log_infos(self, losses, epoch, iter, eta, lr, best_score):
        log_infos = 'Epoch: {}, Iter: {}, ETA: {:.2f} hours, lr: {:.6f}, Best_score: {:.2f} |'.format(
            epoch, iter, eta / 3600, lr, best_score)
        for k, v in losses.items():
            if not k == 'loss':
                log_infos += ' {}: {:.5f},'.format(k, v.numpy()[0])
        log_infos += ' loss: {:.5f}'.format(losses['loss'].numpy()[0])
        return log_infos

    def _show_cue(self, imgs, label, cue, i):
        _, c, h, w = imgs.shape
        cues = np.zeros((h, 2*w, 3), dtype=np.uint8)
        ll = label.numpy()[i, 0]
        img = imgs.numpy()[i, ...].transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        cc = cue[i, ...].transpose(1, 2, 0)
        cc = (cc - cc.min()) / (cc.max() - cc.min()) * 255
        cues[:, 0:w, :] = img.astype(np.uint8)
        cues[:, w:, :] = cc.astype(np.uint8)
        cue_dir = os.path.join(self.checkpoint_config['work_dir'], 'cue')
        if not os.path.exists(cue_dir):
            os.makedirs(cue_dir)
        cv2.imwrite(os.path.join(cue_dir, '{}_imgcue_label_{}.png'.format(i, ll)), cues)

    def _data_to_variable(self, datas):
        imgs = np.array([data[0] for data in datas]).astype(np.float32)
        mask = np.array([data[1] for data in datas]).astype(np.float32)
        label = np.array([data[2] for data in datas]).astype(np.int64).reshape(-1, 1)
        imgs = fluid.dygraph.to_variable(imgs)
        mask = fluid.dygraph.to_variable(mask)
        label = fluid.dygraph.to_variable(label)
        mask.stop_gradient = True
        label.stop_gradient = True
        return imgs, mask, label

    def test(self, is_show=False, thr='auto'):
        results = []
        place = fluid.CUDAPlace(0)
        
        with fluid.dygraph.guard(place):
            self.init_model()
            self.model.eval()
            test_loader = paddle.batch(self.dataset.test(), batch_size=self.val_batch_size, drop_last=False)
            for iter, datas in enumerate(test_loader()):
                batch_size = len(datas)
                imgs = np.array([data[0] for data in datas]).astype(np.float32)
                label = np.array([data[1] for data in datas]).astype(np.int64).reshape(-1, 1)
                imgs = fluid.dygraph.to_variable(imgs)
                label = fluid.dygraph.to_variable(label)
                label.stop_gradient = True
                
                cue = self.model(imgs, label, return_loss=False)
                for i in range(batch_size):
                    score = 1 - cue[i, ...].mean()
                    results.append([score, label.numpy()[i, 0]])

                    if is_show:
                        self._show_cue(imgs, label, cue, i)        

        score = eval_metric(results, thr=thr, type=self.eval_type, res_dir=self.checkpoint_config['work_dir'])
        self.logger.info('Best {}:{:.2f}'.format(self.eval_type, score))

    def val(self):
        results = []
        self.model.eval()
        test_loader = paddle.batch(self.val_dataset.test(), batch_size=self.val_batch_size, drop_last=False)
        for iter, datas in enumerate(test_loader()):
            batch_size = len(datas)
            imgs = np.array([data[0] for data in datas]).astype(np.float32)
            label = np.array([data[1] for data in datas]).astype(np.int64).reshape(-1, 1)
            imgs = fluid.dygraph.to_variable(imgs)
            label = fluid.dygraph.to_variable(label)
            label.stop_gradient = True

            cue = self.model(imgs, label, return_loss=False)
            for i in range(batch_size):
                score = 1 - cue[i, ...].mean()
                results.append([score, label.numpy()[i, 0]])
        self.model.train()
        return eval_metric(results, thr='mid', type=self.eval_type)

    def train(self, max_epochs):
        best_iter = 0
        best_score = 0
        place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) if self.multi_gpus else fluid.CUDAPlace(0)
        with fluid.dygraph.guard(place):
            train_loader = paddle.batch(self.dataset.train(), batch_size=self.batch_size, drop_last=True)
            #train_loader = self.dataset.data_loader(batch_size=self.batch_size, place=place)
            iter_per_epoch = int(self.dataset.length()/self.batch_size)

            self.init_model()
            if self.multi_gpus:
                train_loader = fluid.contrib.reader.distributed_batch_reader(train_loader)
                strategy = fluid.dygraph.parallel.prepare_context()
                self.model = fluid.dygraph.parallel.DataParallel(self.model, strategy)

            optimizer = self.init_optimizer()
            self.logger.info('Total epoch: {}, Total iter: {}, Iter/epoch: {}'.format(
                max_epochs, max_epochs * iter_per_epoch, iter_per_epoch))
            start_time = time.time()
            iter_times = 0
            for epoch in range(max_epochs):
                for iter, datas in enumerate(train_loader()):
                    iter_times += time.time() - start_time
                    start_time = time.time()
                    iter = epoch * iter_per_epoch + iter

                    save_parameters = (not self.multi_gpus) or (
                            self.multi_gpus and fluid.dygraph.parallel.Env().local_rank == 0)

                    if iter % self.checkpoint_config['eval_interval'] == 0 and save_parameters:
                        score = self.val()
                        if score > best_score:
                            best_score = score
                            best_iter = iter
                            self.save_checkpoint(epoch, iter, is_best=True)

                    imgs, mask, label = self._data_to_variable(datas)

                    losses = self.model(imgs, label, mask=mask)
                    loss = losses['loss']

                    if self.multi_gpus:
                        loss = self.model.scale_loss(loss)
                        loss.backward()
                        self.model.apply_collective_grads()
                    loss.backward()
                    
                    optimizer.minimize(loss)
                    self.model.clear_gradients() 

                    if iter % self.log_interval == 0:
                        lr = optimizer.current_step_lr()
                        iters = 1 if iter == 0 else self.log_interval
                        eta = (max_epochs * iter_per_epoch - iter) * iter_times / iters
                        iter_times = 0
                        self.logger.info(self._log_infos(losses, epoch, iter, eta, lr, best_score))

                    if iter and iter % self.checkpoint_config['save_interval'] == 0 and save_parameters:
                        self.save_checkpoint(epoch, iter)

            self.logger.info('Best model save from iter {}'.format(best_iter))



