import numpy as np
import caffe
import image_handling
import matplotlib.pyplot as plt
from gco_python import pygco

class DepthPredictor():
   def __init__(self):
      self.unaryCNN=None

   def load(self,model_file=None,pretrained_file=None, meanfile=None, image_dims=(227,227)):
      """ Load pretrained classifiers
            model_file = model descripton file (*.prototxt)
            pretrained_file = pretrained weights (*.caffemodel)
            mean = 2D array of training data means
      """
      # Instantiate net

      self.unaryCNN = caffe.Net(model_file,pretrained_file)

      # Params (Likely constant for image sets)
      self.unaryCNN.set_raw_scale(self.unaryCNN.inputs[0], 255)
      # order of channels: RGB, BGR, etc? 

      #self.unaryCNN.set_channel_swap(self.unaryCNN.inputs[0], (2,1,0))
      #print self.unaryCNN.inputs
      # mean file?
      if meanfile is None:
         self.mean = np.zeros((image_dims[0],image_dims[1],3))
      else:
         self.mean = 255* caffe.io.resize_image(meanfile, image_dims)

      #    print "Mean is: ",meanfile.shape
      #    mean_resize = caffe.io.resize_image(meanfile, image_dims).astype('uint8')
      #    print "Mean_resize is: ",mean_resize.shape
      #    self.unaryCNN.set_mean(self.unaryCNN.inputs[0], mean_resize.transpose(2,1,0))

      #self.unaryCNN.crop_dims = np.array(self.unaryCNN.blobs[self.unaryCNN.inputs[0]].data.shape[2:])
      self.unaryCNN.image_dims = image_dims


   def predict(self, input_img, no_superpixels=400, true_depths=None, graphcut=False, single_mode=False):
      """ Predict a depth field """
      #return self.unaryCNN.predict(images)
      # Scale to standardize input dimensions.
      oversample = False   # Flip it, shift it, crop it, etc
      
      input_segments = []
      self.window_size = 83  # 167x167
      if not single_mode:
         print "image dims: ",self.unaryCNN.image_dims
         # Preprocess image: should return segment patches, depth map, etc.
         [segs, mask] = image_handling.preprocess_image(input_img, true_depth=None,
               no_superpixels=no_superpixels, x_window_size=self.window_size, y_window_size=self.window_size)
      
         print "seg length = ", len(segs)
         input_ = np.zeros((len(segs),self.unaryCNN.image_dims[0], self.unaryCNN.image_dims[1],3),dtype=np.float32)

      
         # Resize & subtract mean
         for ix, in_ in enumerate(segs):
   #         input_[ix] = caffe.io.resize_image(in_.transpose(2,1,0), self.unaryCNN.image_dims).astype('uint8') #- self.mean

            input_[ix] = caffe.io.resize_image(in_.transpose(2,1,0), self.unaryCNN.image_dims) - self.mean

      else: # just classifying one frame
         segs = input_img
         tmp_img = caffe.io.resize_image(input_img, self.unaryCNN.image_dims) - self.mean
         input_= np.zeros([1,tmp_img.shape[0],tmp_img.shape[1],tmp_img.shape[2]])
         input_[0,:,:,:] = tmp_img
      #print "max inp: ",np.max(input_[1])
      #print "max avg: ",np.max(self.mean)
      # print segs[1,:,:,:].shape
      # print input_[1,:,:,:].shape
      # plt.subplot(211)
      # plt.imshow(segs[1,:,:,:].transpose(2,1,0))
      # plt.subplot(212)
      # plt.imshow(input_[1,:,:,:])
      # plt.show()
      



      # Classify
      # Allocate memory
      caffe_in = np.zeros(np.array(input_.shape)[[0,3,1,2]],
                         dtype=np.float32)

      # Preprocess 
      for ix, in_ in enumerate(input_):
         caffe_in[ix] = self.unaryCNN.preprocess(self.unaryCNN.inputs[0], in_)

      # Run through CNN
      out = self.unaryCNN.forward_all(**{self.unaryCNN.inputs[0]: caffe_in})
     
      #print "inp label is: ",self.unaryCNN.inputs[0]
      # Sum over R, G, B layers (kept separate?)
      predictions = out[self.unaryCNN.outputs[0]].squeeze(axis=(2,3))


      if graphcut:
         print "heyo"

      else:
         out = self.unaryCNN.forward_all(**{self.unaryCNN.inputs[0]: caffe_in})


         #print "inp label is: ",self.unaryCNN.inputs[0]
         predictions = out[self.unaryCNN.outputs[0]].squeeze(axis=(2,3))
         #plt.imshow(predictions)
         #plt.show()
         #print np.argmax(predictions,axis=1)

      #if not single_mode:
         # Softmax output
         #print predictions
         #return image_handling.apply_depths(np.argmax(predictions,axis=1), mask), predictions, segs, mask

         # Regression output
      return image_handling.apply_depths(predictions, mask), predictions, segs, mask
      #else:
      #   return predictions

   def train(self,solver_path=None):
      """
       Trains the convnet, using the solver defined in solver_path.prototxt
      """
      if self.unaryCNN==None:
         print 'Please load a model file first!'


      S = SGDSolver(solver_path)

      print type(S)



