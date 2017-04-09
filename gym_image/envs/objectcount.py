import gym
from gym import spaces
import skimage.io, skimage.transform, skimage.measure, skimage.exposure, skimage.feature
import numpy as np
import keras
import glob
import re

CLSCOLOR = ((243,8,5),
            (244,8,242),
            (87,46,10),
            (25,56,176),
            (38,174,21))

CLSAREA = (20, 20, 28, 18, 20)

IMH, IMW = (224, 224)

def _get_extractor(mod = "VGG16"):
    model, fshape, preproc = (None, None, None)
    if mod == "VGG16":
        print("Using VGG16 as feature extractor")
        model = keras.applications.vgg16.VGG16(False)
        fshape = 7 * 7 * 512
        preproc = keras.applications.vgg16.preprocess_input
    elif mod == "VGG19":
        print("Using VGG19 as feature extractor")
        model = keras.applications.vgg19.VGG19(False)
        fshape = 7 * 7 * 512
        preproc = keras.applications.vgg19.preprocess_input
    elif mod == "resnet50":
        print("Using resnet50 as feature extractor")
        model = keras.applications.resnet50.ResNet50(False)
        fshape = 2048
        preproc = keras.applications.resnet50.preprocess_input
    return(model, fshape, preproc)
    
class ObjectCountData(object):
    def __init__(self, ids = None, img_path = "../input/Train/", lab_path = "../input/TrainDotted/",
                 remove_bg = True, equalize = True, gamma = False, intensity = True, log = False,
                 thresh = 15/255):
        if ids is None:
            ids = glob.glob("../input/Train/*.jpg")
            for iid, id in enumerate(ids):
                ids[iid] = re.search(".*\\\\(.+?).jpg",str(id)).group(1)
        ids = [0,1]
        self.N = len(ids)
        self.ids = ids
        self.img_path = img_path
        self.lab_path = lab_path
        self.remove_bg = remove_bg
        self.equalize = equalize
        self.gamma = gamma
        self.intensity = intensity
        self.log = log
        self.images = []
        for id in ids:
            self.images.append(self.get_info(id, thresh=thresh))
            
    def get_info(self, id, thresh = 15/255):
        img = skimage.io.imread(self.img_path + str(id) + ".jpg")
        lab = skimage.io.imread(self.lab_path + str(id) + ".jpg")
        if self.remove_bg:
            dif = abs(img/255 - lab/255) > thresh
            dif[:,:,0] = dif.max(2)
            dif[:,:,1] = dif.max(2)
            dif[:,:,2] = dif.max(2)
            lab = dif * lab
        if self.intensity:
            img = skimage.exposure.rescale_intensity(img)
        if self.equalize:
            img = skimage.exposure.equalize_hist(img)
        if self.log:
            img = skimage.exposure.adjust_log(img)
        if self.gamma:
            img = skimage.exposure.adjust_gamma(img)
        return(ObjectCountImage(img, lab))

class ObjectCountImage(object):
    def __init__(self, image, label=None, count = False):
        self.image = image
        self.label = label
        if label is not None:
            assert(self.image.shape == self.label.shape)
        self.imw = image.shape[1]
        self.imh = image.shape[0]
        if count:
            self.count = self.count_label()
        else:
            self.count = None
        
    def subimage(self, pos = None, border = 0, minsize = 50):
        if pos is None:
            if border == 0:
                return(self)
            else:
                if self.label is None:
                    return(ObjectCountImage(self.image[(0 + border):(self.imh - border), (0 + border):(self.imw - border)]))
                else:
                    return(ObjectCountImage(self.image[(0 + border):(self.imh - border), (0 + border):(self.imw - border)], self.label[(0 + border):(self.imh - border), (0 + border):(self.imw - border)]))
        else:
            assert(len(pos) == 4)
            if pos[2] - pos[0] < 50:
                pos[2] = pos[0] + 50
            if pos[3] - pos[1] < 50:
                pos[3] = pos[1] + 50
            if self.label is None:
                return(ObjectCountImage(self.image[max(0, pos[0]):min(self.imh, pos[2]), max(0, pos[1]):min(self.imw, pos[3])]))
            else:
                return(ObjectCountImage(self.image[max(0, pos[0]):min(self.imh, pos[2]), max(0, pos[1]):min(self.imw, pos[3])], self.label[max(0, pos[0]):min(self.imh, pos[2]), max(0, pos[1]):min(self.imw, pos[3])]))
                
    def show_image(self):
        skimage.io.imshow(self.image)
    
    def show_label(self):
        if self.label is not None: skimage.io.imshow(self.label)
    
    def draw_region(self, pos = None, border = 0):
        tmpimg = self.image
        tmplab = self.label
        if pos is not None:
            if border > 0:
                pos = (pos[0] + border, pos[1] + border, pos[2] - border, pos[3] - border)
            tmpimg[pos[0]:pos[2],pos[1]:pos[3]] = [255,0,0]
            tmplab[pos[0]:pos[2],pos[1]:pos[3]] = [255,0,0]
            tmpimg[(pos[0] + 3):(pos[2] - 3),(pos[1] + 3):(pos[3] - 3)] = self.image[(pos[0] + 3):(pos[2] - 3),(pos[1] + 3):(pos[3] - 3)]
            tmplab[(pos[0] + 3):(pos[2] - 3),(pos[1] + 3):(pos[3] - 3)] = self.label[(pos[0] + 3):(pos[2] - 3),(pos[1] + 3):(pos[3] - 3)]
            return(ObjectCountImage(tmpimg, tmplab))
        return(self)

    def mark_region(self, pos = None, border = 0):
        if pos is not None:
            if border > 0:
                pos = (pos[0] + border, pos[1] + border, pos[2] - border, pos[3] - border)
            self.image[pos[0]:pos[2],pos[1]:pos[3],:] = 0
            self.label[pos[0]:pos[2],pos[1]:pos[3],:] = 0
                          
    def count_label(self, thresh = 32, border = 0):
        if self.label is not None:
            label = self.label
            if border > 0:
                label = label[border:(label.shape[0] - border), border:(label.shape[1] - border)]
            nn = []
            for cci in range(len(CLSCOLOR)):
                labimg, labcnt = skimage.measure.label(np.sqrt(np.sum(np.square(label - CLSCOLOR[cci]), axis = -1)) < thresh, neighbors = 8, return_num = True)
                labreg = skimage.measure.regionprops(labimg)
                for prop in labreg:
                    if prop.area < CLSAREA[cci] * 0.3:
                        labcnt -= 1
                    elif prop.area > CLSAREA[cci] * 1.7:
                        labcnt += int(prop.area / CLSAREA[cci])
                nn.append(labcnt)
            return(nn)
        else:
            return(None)


class ObjectCountEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, ids = None, nclass = 5, nhist = 10, alpha = 0.2, border = 20, extractor = "resnet50"):
        if ids is None:
            ids = [0,1,15]
        self.ids = ids
        self.images = ObjectCountData(self.ids)
        self.img_count = None
        
        self.nclass = nclass
        self.nhist = nhist
        self.alpha = alpha
        self.border = border
        self.model, self.fshape, self.preproc = _get_extractor(extractor)
        
        self.action_space = spaces.Discrete(10 + self.nclass)
        self.observation_space = spaces.Box(np.zeros((self.nhist * (10 + self.nclass) + self.fshape)), np.append(np.zeros((self.nhist * (10 + self.nclass))), np.ones((self.fshape)) * 100))
        self._reset()
        self._reset_image()
    
    def _step(self, action):
        if self.done:
            return(self.state, self.reward, self.done, {'state': self.state})
        hup = np.zeros((10 + self.nclass))
        hup[action] = 1
        self._take_action(action)
        self.state = self._get_state(image = self.images.images[self.imgid].subimage(self.bbox).image, history = hup)
        return(self.state, self.reward, self.done, {'state': self.state})
        
    def _reset(self):
        self.done = False
        self.imgid = -1
        self.reward = 0
        self.nstep = 0
        self.history = np.zeros((self.nhist, 10 + self.nclass))
        self.state = self._get_state()
        return(self.state)
        
    def _reset_image(self):
        self.imgid += 1
        if self.imgid + 1 >= self.images.N:
            self.done = True
            return
        self.bbox = (0, 0, self.images.images[self.imgid].imh, self.images.images[self.imgid].imw)
        self.img_reward = 0
        self.img_count = np.zeros((self.nclass))
        self.reg_count = np.zeros((self.nclass))
        self.curr_reward = 0
        self.prev_reward = 0
        
    def _render(self, mode='human', close=False):
        print("Counts")
        print(self.img_count)
        return(self.img_count)
        
    def _get_state(self, image = None, history = None):
        if image is None:
            feat = np.zeros(self.fshape)
        else:
            print(image.shape)
            image = skimage.transform.resize(image, (IMW, IMH, 3))
            feat = self.model.predict(self.preproc(np.expand_dims(image, axis=0)))
        hist = self.history
        if history is not None:
            hist[0:-1] = hist[1:]
            hist[-1] = history
        
        return(np.append(hist.flatten(), feat))
    
    def _get_reward(self):
        img_count = self.images.images[self.imgid].count_label(border = self.border)
        return(self.img_reward * np.max(np.abs(self.img_count - img_count) / img_count))
        
    def _get_reg_reward(self):
        self.curr_reward = 1/(np.sum(np.abs(self.reg_count - self.images.images[self.imgid].subimage(self.bbox).count_label(border = self.border))) + 0.0001)
        if self.curr_reward > self.prev_reward:
            return(1)
        else:
            return(0)
        
    def _take_action(self, action):
        """
        Actions:
            0. Move left
            1. Move right
            2. Move up
            3. Move down
            4. Increase width
            5. Decrease width
            6. Increase height
            7. Decrease height
            8. Mark Counted
            9. Next image
            10+ Add class count
        """
        xrng = self.bbox[3] - self.bbox[1]
        yrng = self.bbox[2] - self.bbox[0]
        xshft = self.alpha * xrng
        yshft = self.alpha * yrng
        if action < 8:
            if xrng < 50:
                xshft = 0
            if yrng < 50:
                yshft = 0
        if action == 0:
            self.bbox = (self.bbox[0], self.bbox[1] - xshft, self.bbox[2], self.bbox[3] - xshft)
        elif action == 1:
            self.bbox = (self.bbox[0], self.bbox[1] + xshft, self.bbox[2], self.bbox[3] + xshft)
        elif action == 2:
            self.bbox = (self.bbox[0] - yshft, self.bbox[1], self.bbox[2] - yshft, self.bbox[3])
        elif action == 3:
            self.bbox = (self.bbox[0] + yshft, self.bbox[1], self.bbox[2] + yshft, self.bbox[3])
        elif action == 4:
            self.bbox = (self.bbox[0], self.bbox[1] - xshft, self.bbox[2], self.bbox[3] + xshft)
        elif action == 5:
            self.bbox = (self.bbox[0], self.bbox[1] + xshft, self.bbox[2], self.bbox[3] - xshft)
        elif action == 6:
            self.bbox = (self.bbox[0] - yshft, self.bbox[1], self.bbox[2] + yshft, self.bbox[3])
        elif action == 7:
            self.bbox = (self.bbox[0] + yshft, self.bbox[1], self.bbox[2] - yshft, self.bbox[3])
        elif action == 8:
            reg_r = self._get_reg_reward()
            self.img_reward += reg_r
            self.reward += reg_r
            self.images.images[self.imgid].mark_region(pos = self.bbox, border = self.border)
            self.reg_count = np.zeros((self.nclass))
            self.bbox = (0,0,self.images.images[self.imgid].imh,self.images.images[self.imgid].imw)
        elif action == 9:
            self.reward += self._get_reward()
            self._reset_image()
        else:
            self.img_count[action - 10] += 1
            self.reg_count[action - 10] += 1
            

            
            
            
