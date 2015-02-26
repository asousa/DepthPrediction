import numpy as np
import caffe

class DepthPredictor:
	def __init__(self):
		self.singleCNN=None

	def load(self,model_file=None,pretrained_file=None,meanfile=None):
		""" Load pretrained classifiers
				model_file = model descripton file (*.prototxt)
				pretrained_file = pretrained weights (*.caffemodel)
				mean = 2D array of training data means (I think)
		"""
		#dims = meanfile.shape
		#print dims[:2]
		self.singleCNN=caffe.Classifier(model_file, pretrained_file,mean=meanfile,
										channel_swap=(2,1,0),raw_scale=255, image_dims=(227,227),
										gpu=True)
#										channel_swap=(2,1,0), raw_scale=255, image_dims=(256, 256))


	def predict(self,images):
		""" Predict a depth field (OK, currently just classifying images) """
		return self.singleCNN.predict(images)

