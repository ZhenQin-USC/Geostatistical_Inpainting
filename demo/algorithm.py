from scipy.optimize import Bounds, minimize
import tensorflow as tf
import numpy as np
import copy
import random


class One_patch_inpainting:

    def __init__(self, ytrue, mask, wass_gan, x0=None, z_dim=100, lam=10):
        self.ytrue = ytrue
        self.mask = mask
        self.wgan = wass_gan
        self.z_dim = z_dim
        self.lam = lam
        self.wgan.generator.trainable = False
        self.wgan.critic.trainable = False
        self.c_loss = None
        self.p_loss = None
        self.t_loss = None
        self.HistoryFval = {'c_loss': [], 'p_loss': [], 't_loss': []}
        self.HistoryX = []
        if x0 is None:
            self.noise = np.random.normal(0, 1, (self.z_dim))
        else:
            self.noise = x0

    def context_loss(self, ypred):
        _ = tf.keras.layers.Flatten()(tf.abs(tf.multiply(self.mask, ypred) - tf.multiply(self.mask, self.ytrue)))
        # print(_)
        c_loss = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis=-1))(_)
        return c_loss

    def gradfun(self, noise):

        noise = tf.Variable(noise[None])
        with tf.GradientTape() as tape:
            tape.watch(noise)
            image = self.wgan.generator(noise)

            c_loss = self.context_loss(image)
            p_loss = self.wgan.critic(image)
            t_loss = self.lam * p_loss + c_loss

            grad = tape.gradient(t_loss, noise)[0]
            return grad

    def objfun(self, noise):
        noise = noise[None].astype('float32')
        image = self.wgan.generator(noise)
        self.c_loss = self.context_loss(image)
        self.p_loss = self.wgan.critic(image)
        self.t_loss = self.lam * self.p_loss + self.c_loss
        return self.t_loss.numpy()[0]  # , c_loss.numpy(), p_loss.numpy()

    def optimizer(self, x0=None):
        if x0 is None:
            x0 = self.noise
        res = minimize(self.objfun, x0,
                       method='BFGS', jac=self.gradfun, callback=self.savedata,
                       options={'gtol': 1e-05, 'eps': 1e-08,
                                'maxiter': 2e2, 'disp': True})
        return res, self.HistoryFval, self.HistoryX

    def savedata(self, x):
        history_c_loss = self.HistoryFval['c_loss']
        history_p_loss = self.HistoryFval['p_loss']
        history_t_loss = self.HistoryFval['t_loss']

        history_c_loss.append(self.c_loss.numpy())
        history_p_loss.append(self.p_loss.numpy())
        history_t_loss.append(self.t_loss.numpy())

        self.HistoryFval['c_loss'] = history_c_loss
        self.HistoryFval['p_loss'] = history_p_loss
        self.HistoryFval['t_loss'] = history_t_loss

        self.HistoryX.append(x)

    def inpainting(self, thresh=1e-4, totalEpoch=20):
        i = 0
        res, self.HistoryFval, self.HistoryX = self.optimizer()
        while i < totalEpoch and res['fun'] > thresh:
            i += 1
            print('Re-sampling of z0: No.{}'.format(i))
            x0 = np.random.normal(0, 1, (100))
            self.HistoryFval = {'c_loss': [], 'p_loss': [], 't_loss': []}
            self.HistoryX = []
            res, self.HistoryFval, self.HistoryX = self.optimizer(x0)
        return res, self.HistoryFval, self.HistoryX


class Sequential_inpainting:

    def __init__(self, whole_map, ndata, wass_gan, masks=None, nbx=3, nby=3, ngx=28, ngy=28, bandwidth=2):
        """
        whole_map: 2D-array of ground truth with dimension of NX-by-NY
        NX, NY: the dimension of the whole map
        ndata: number of hard data points
        wass_gan: wasserstein-gan model
        nbx, nby: block dimensions in x- and y-directions
        ngx, ngy: grid dimensions in x- and y-directions
        bandwidth: the allowed width of overlapped patch
        """
        # main input
        self.whole_map = whole_map # 2D-array: NX-by-NY
        NX, NY = whole_map.shape
        self.NX = NX
        self.NY = NY
        self.ndata = ndata
        self.wgan = wass_gan

        # default input
        self.nbx = nbx
        self.nby = nby
        self.ngx = ngx
        self.ngy = ngy
        self.bandwidth = bandwidth

        # create mask for the whole map
        if masks is None:
            self.masks = self.gen_mask()
        else:
            self.masks = masks

        # create an empth map with unknown grid as nan
        self.empty_map = np.asarray(copy.deepcopy(whole_map),dtype='float32') # 2D-array: NX-by-NY
        self.empty_map[self.masks[0,:,:]==0] = np.nan

        # create one random path
        self.random_path = self.gen_randompath(npath = 1)

        # initialization
        self.ytrue = None # current local patch of ground truth
        self.mask = None # current local path of masks
        self.res = []
        self.HistoryFval = []
        self.HistoryX = []
        self.mask_record = []
        self.ytrue_record = []

    def simulate(self):

        # given empty_map, random_path,
        for loc in self.random_path[0]:
            # get current local ground truth and mask
            self.mask, _, self.ytrue, xi, yi = self.get_localmask(loc)
            worker = One_patch_inpainting(self.ytrue, self.mask, self.wgan)
            res, HistoryFval, HistoryX = worker.inpainting(thresh=1e-3, totalEpoch=40)

            # get optimal image
            image = self.wgan.generator(res['x'][None])

            # update empty map
            self.empty_map[xi:xi+self.ngx, yi:yi+self.ngy] = image

            # save intermediate results
            self.res.append(res)
            self.HistoryFval.append(HistoryFval)
            self.HistoryX.append(HistoryX)
            self.mask_record.append(self.mask)
            self.ytrue_record.append(self.ytrue)

    def gen_mask(self):
        masks = np.zeros((1,self.NX,self.NY)).astype('float32')
        mask_ind = []
        mask_ind.append(random.sample(range(self.NX*self.NY), self.ndata))
        index = np.unravel_index(mask_ind[0], (self.NX,self.NY))
        masks[0, index[0],index[1]] = 1
        return masks

    def gen_randompath(self, npath = 10):
        # npath: number of random path to generate
        nblock = self.ngx * self.ngy
        random_paths = []
        i = 0
        while i < npath:
            path_ = np.random.permutation(nblock) # generate one random path
            if i >= 1:
                if any((path_ - np.array(random_paths)).max(axis=1) == 0): # repeated path --> re-sample random path
                    pass
                else: # sample accepted
                    random_paths.append(path_)
                    i += 1
            else:
                random_paths.append(path_)
                i += 1
        return np.asarray(random_paths)

    def get_localmask(self, loc):
        # loc: index or location of the current patch
        block_index = np.unravel_index(loc, shape=(self.nbx,self.nby)) # block dimension: 3-by-3 by default
        mask = np.ones((self.ngx,self.ngy)) # grid dimension: 28-by-28 by default
        xi = block_index[0]*self.ngx-block_index[0]*self.bandwidth # initial index in x-direction
        yi = block_index[1]*self.ngx-block_index[1]*self.bandwidth # initial index in y-direction
        slice_data = self.empty_map[xi:xi+self.ngx, yi:yi+self.ngy] # local patch of the empty map
        ytrue = self.whole_map[xi:xi+self.ngx, yi:yi+self.ngy] # local patch of the empty map
        mask[np.isnan(slice_data)] = 0
        return mask, slice_data, ytrue, xi, yi