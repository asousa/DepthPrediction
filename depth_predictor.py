import numpy as np
import caffe
from unaryRegressor import Regressor

class DepthPredictor():
   def __init__(self):
      self.unaryCNN=None

   def load(self,model_file=None,pretrained_file=None, meanfile=None, image_dims=(256,256)):
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
      self.unaryCNN.set_channel_swap(self.unaryCNN.inputs[0], (2,1,0))

      # mean file?
      if meanfile is not None:
         self.unaryCNN.set_mean(unaryCNN.inputs[0], meanfile)

      self.unaryCNN.crop_dims = np.array(self.unaryCNN.blobs[self.unaryCNN.inputs[0]].data.shape[2:])
      self.unaryCNN.image_dims = image_dims


   def predict(self,inputs):
      """ Predict a depth field (OK, currently just classifying images) """
      #return self.unaryCNN.predict(images)
      # Scale to standardize input dimensions.
      oversample = True
      
      input_ = np.zeros((len(inputs),
         self.unaryCNN.image_dims[0], self.unaryCNN.image_dims[1], inputs[0].shape[2]),
         dtype=np.float32)
      
      for ix, in_ in enumerate(inputs):
         input_[ix] = caffe.io.resize_image(in_, self.unaryCNN.image_dims)

      # Generate center, corner, and mirrored crops.
      input_ = caffe.io.oversample(input_, self.unaryCNN.crop_dims)
      
      # Classify
      caffe_in = np.zeros(np.array(input_.shape)[[0,3,1,2]],
                         dtype=np.float32)
      for ix, in_ in enumerate(input_):
         caffe_in[ix] = self.unaryCNN.preprocess(self.unaryCNN.inputs[0], in_)
      out = self.unaryCNN.forward_all(**{self.unaryCNN.inputs[0]: caffe_in})
      predictions = out[self.unaryCNN.outputs[0]].squeeze(axis=(2,3))

      # For oversampling, average predictions across crops.
      if oversample:
         predictions = predictions.reshape((len(predictions) / 10, 10, -1))
         predictions = predictions.mean(1)

      return predictions