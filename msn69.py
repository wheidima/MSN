import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv3D
from keras.layers import MaxPooling3D
from keras.layers import AveragePooling3D
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Lambda
from keras.layers import GlobalAveragePooling3D
from keras import backend as K

def conv3d_bn(x,filters,num_frames,num_row,num_col,padding='same',strides=(1,1,1),use_bias=False,use_activation_fn=True,use_bn=True,name=None):

	if name is not None:
		bn_name = name + '_bn'
		conv_name = name + '_conv'
	else:
		bn_name = None
		conv_name = None

	x = Conv3D(filters,(num_frames,num_row,num_col),strides=strides,padding=padding,use_bias=use_bias,name=conv_name)(x)
	
	if use_bn:
		if K.image_data_format() == 'channels_first':
			bn_axis = 1
		else:
			bn_axis = 4
		x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

	if use_activation_fn:
		x = Activation('relu', name=name)(x)

	return x


def MSN(include_top=True,weights=None,input_tensor=None,input_shape=None,dropout_prob=0.0,endpoint_logit=True,classes=8631):
	
	if input_tensor is None:
		img_input = Input(shape=input_shape)
	else:
		if not K.is_keras_tensor(input_tensor):
			img_input = Input(tensor=input_tensor,shape=input_shape)
		else:
			img_input = input_tensor


	if K.image_data_format() == 'channels_first':
		channel_axis = 1
	else:
		channel_axis = 4


	#Downsampling via convolution (spatial and temporal)
	x = conv3d_bn(img_input,64,7,7,7,strides=(1,2,2),padding='same',name='Conv3d_1_7x7x3')

	# Downsampling (spatial only)
	x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same', name='MaxPool2d_1_2x2')(x)

	#Conv2 Layer
	#Stage 1
	branch_0 = conv3d_bn(x,128,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_2_s1_loop')

	branch_1 = conv3d_bn(x,32,3,1,1,padding='same',name='Conv3d_2_s1_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,32,3,3,3,padding='same',name='Conv3d_2_s1_3x3x3')
	branch_3 = conv3d_bn(branch_1,32,5,5,5,padding='same',name='Conv3d_2_s1_5x5x5')
	branch_4 = conv3d_bn(branch_1,32,7,7,7,padding='same',name='Conv3d_2_s1_7x7x7')

	branch_1 = layers.concatenate([branch_2,branch_3,branch_4],axis=channel_axis,name='Conv3d_2_s1_conca')

	branch_1 = conv3d_bn(branch_1,128,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_2_s1_F')

	x = layers.add([branch_0, branch_1])

	x = Activation('relu', name='Conv3d_2_s1_F_Relu')(x)
	
	#Stage 2
	branch_0 = conv3d_bn(x,128,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_2_s2_loop')

	branch_1 = conv3d_bn(x,32,3,1,1,padding='same',name='Conv3d_2_s2_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,32,3,3,3,padding='same',name='Conv3d_2_s2_3x3x3')
	branch_3 = conv3d_bn(branch_1,32,5,5,5,padding='same',name='Conv3d_2_s2_5x5x5')
	branch_4 = conv3d_bn(branch_1,32,7,7,7,padding='same',name='Conv3d_2_s2_7x7x7')

	branch_1 = layers.concatenate([branch_2,branch_3,branch_4],axis=channel_axis,name='Conv3d_2_s2_conca')

	branch_1 = conv3d_bn(branch_1,128,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_2_s2_F')

	x = layers.add([branch_0, branch_1])
	
	x = Activation('relu', name='Conv3d_2_s2_F_Relu')(x)


	#Stage 3
	branch_0 = conv3d_bn(x,128,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_2_s3_loop')

	branch_1 = conv3d_bn(x,32,3,1,1,padding='same',name='Conv3d_2_s3_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,32,3,3,3,padding='same',name='Conv3d_2_s3_3x3x3')
	branch_3 = conv3d_bn(branch_1,32,5,5,5,padding='same',name='Conv3d_2_s3_5x5x5')
	branch_4 = conv3d_bn(branch_1,32,7,7,7,padding='same',name='Conv3d_2_s3_7x7x7')

	branch_1 = layers.concatenate([branch_2,branch_3,branch_4],axis=channel_axis,name='Conv3d_2_s3_conca')

	branch_1 = conv3d_bn(branch_1,128,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_2_s3_F')

	x = layers.add([branch_0, branch_1])
	
	x = Activation('relu', name='Conv3d_2_s3_F_Relu')(x)

	#Conv3 Layer
	#Stage 1
	branch_0 = conv3d_bn(x,256,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_3_s1_loop')

	branch_1 = conv3d_bn(x,64,3,1,1,padding='same',name='Conv3d_3_s1_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,64,3,3,3,padding='same',name='Conv3d_3_s1_3x3x3')
	branch_3 = conv3d_bn(branch_1,64,5,5,5,padding='same',name='Conv3d_3_s1_5x5x5')
	branch_4 = conv3d_bn(branch_1,64,7,7,7,padding='same',name='Conv3d_3_s1_7x7x7')

	branch_1 = layers.concatenate([branch_2,branch_3,branch_4],axis=channel_axis,name='Conv3d_3_s1_conca')

	branch_1 = conv3d_bn(branch_1,256,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_3_s1_F')

	x = layers.add([branch_0, branch_1])

	x = Activation('relu', name='Conv3d_3_s1_F_Relu')(x)

	#Stage 2
	branch_0 = conv3d_bn(x,256,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_3_s2_loop')

	branch_1 = conv3d_bn(x,64,3,1,1,padding='same',name='Conv3d_3_s2_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,64,3,3,3,padding='same',name='Conv3d_3_s2_3x3x3')
	branch_3 = conv3d_bn(branch_1,64,5,5,5,padding='same',name='Conv3d_3_s2_5x5x5')
	branch_4 = conv3d_bn(branch_1,64,7,7,7,padding='same',name='Conv3d_3_s2_7x7x7')

	branch_1 = layers.concatenate([branch_2,branch_3,branch_4],axis=channel_axis,name='Conv3d_3_s2_conca')

	branch_1 = conv3d_bn(branch_1,256,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_3_s2_F')

	x = layers.add([branch_0, branch_1])

	x = Activation('relu', name='Conv3d_3_s2_F_Relu')(x)

	#Stage 3
	branch_0 = conv3d_bn(x,256,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_3_s3_loop')

	branch_1 = conv3d_bn(x,64,3,1,1,padding='same',name='Conv3d_3_s3_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,64,3,3,3,padding='same',name='Conv3d_3_s3_3x3x3')
	branch_3 = conv3d_bn(branch_1,64,5,5,5,padding='same',name='Conv3d_3_s3_5x5x5')
	branch_4 = conv3d_bn(branch_1,64,7,7,7,padding='same',name='Conv3d_3_s3_7x7x7')

	branch_1 = layers.concatenate([branch_2,branch_3,branch_4],axis=channel_axis,name='Conv3d_3_s3_conca')

	branch_1 = conv3d_bn(branch_1,256,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_3_s3_F')

	x = layers.add([branch_0, branch_1])

	x = Activation('relu', name='Conv3d_3_s3_F_Relu')(x)

	#Stage 4
	branch_0 = conv3d_bn(x,256,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_3_s4_loop')

	branch_1 = conv3d_bn(x,64,3,1,1,padding='same',name='Conv3d_3_s4_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,64,3,3,3,padding='same',name='Conv3d_3_s4_3x3x3')
	branch_3 = conv3d_bn(branch_1,64,5,5,5,padding='same',name='Conv3d_3_s4_5x5x5')
	branch_4 = conv3d_bn(branch_1,64,7,7,7,padding='same',name='Conv3d_3_s4_7x7x7')

	branch_1 = layers.concatenate([branch_2,branch_3,branch_4],axis=channel_axis,name='Conv3d_3_s4_conca')

	branch_1 = conv3d_bn(branch_1,256,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_3_s4_F')

	x = layers.add([branch_0, branch_1])

	x = Activation('relu', name='Conv3d_3_s4_F_Relu')(x)

	#Downsampling (spatial and temporal)
	x = MaxPooling3D((2,2,2),strides=(2,2,2),padding='same',name='MaxPoolAfterConv3')(x)

	#Conv4 Layer
	#Stage 1
	branch_0 = conv3d_bn(x,512,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s1_loop')

	branch_1 = conv3d_bn(x,256,3,1,1,padding='same',name='Conv3d_4_s1_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,256,3,3,3,padding='same',name='Conv3d_4_s1_3x3x3')
	
	branch_1 = conv3d_bn(branch_2,512,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s1_F')

	x = layers.add([branch_0, branch_1])

	x = Activation('relu', name='Conv3d_4_s1_F_Relu')(x)

	#Stage 2
	branch_0 = conv3d_bn(x,512,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s2_loop')

	branch_1 = conv3d_bn(x,256,3,1,1,padding='same',name='Conv3d_4_s2_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,256,3,3,3,padding='same',name='Conv3d_4_s2_3x3x3')

	branch_1 = conv3d_bn(branch_2,512,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s2_F')

	x = layers.add([branch_0, branch_1])

	x = Activation('relu', name='Conv3d_4_s2_F_Relu')(x)

	#Stage 3
	branch_0 = conv3d_bn(x,512,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s3_loop')

	branch_1 = conv3d_bn(x,256,3,1,1,padding='same',name='Conv3d_4_s3_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,256,3,3,3,padding='same',name='Conv3d_4_s3_3x3x3')

	branch_1 = conv3d_bn(branch_2,512,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s3_F')

	x = layers.add([branch_0, branch_1])

	x = Activation('relu', name='Conv3d_4_s3_F_Relu')(x)

	#Stage 4
	branch_0 = conv3d_bn(x,512,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s4_loop')

	branch_1 = conv3d_bn(x,256,3,1,1,padding='same',name='Conv3d_4_s4_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,256,3,3,3,padding='same',name='Conv3d_4_s4_3x3x3')

	branch_1 = conv3d_bn(branch_2,512,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s4_F')

	x = layers.add([branch_0, branch_1])
	
	x = Activation('relu', name='Conv3d_4_s4_F_Relu')(x)

	#Stage 5
	branch_0 = conv3d_bn(x,512,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s5_loop')

	branch_1 = conv3d_bn(x,256,3,1,1,padding='same',name='Conv3d_4_s5_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,256,3,3,3,padding='same',name='Conv3d_4_s5_3x3x3')

	branch_1 = conv3d_bn(branch_2,512,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s5_F')

	x = layers.add([branch_0, branch_1])

	x = Activation('relu', name='Conv3d_4_s5_F_Relu')(x)

	#Stage 6
	branch_0 = conv3d_bn(x,512,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s6_loop')

	branch_1 = conv3d_bn(x,256,3,1,1,padding='same',name='Conv3d_4_s6_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,256,3,3,3,padding='same',name='Conv3d_4_s6_3x3x3')

	branch_1 = conv3d_bn(branch_2,512,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s6_F')

	x = layers.add([branch_0, branch_1])
	
	x = Activation('relu', name='Conv3d_4_s6_F_Relu')(x)

	#Stage 7
	branch_0 = conv3d_bn(x,512,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s7_loop')

	branch_1 = conv3d_bn(x,256,3,1,1,padding='same',name='Conv3d_4_s7_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,256,3,3,3,padding='same',name='Conv3d_4_s7_3x3x3')

	branch_1 = conv3d_bn(branch_2,512,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s7_F')

	x = layers.add([branch_0, branch_1])
	
	x = Activation('relu', name='Conv3d_4_s7_F_Relu')(x)

	#Stage 8
	branch_0 = conv3d_bn(x,512,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s8_loop')

	branch_1 = conv3d_bn(x,256,3,1,1,padding='same',name='Conv3d_4_s8_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,256,3,3,3,padding='same',name='Conv3d_4_s8_3x3x3')

	branch_1 = conv3d_bn(branch_2,512,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s8_F')

	x = layers.add([branch_0, branch_1])
	
	x = Activation('relu', name='Conv3d_4_s8_F_Relu')(x)

	#Stage 9
	branch_0 = conv3d_bn(x,512,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s9_loop')

	branch_1 = conv3d_bn(x,256,3,1,1,padding='same',name='Conv3d_4_s9_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,256,3,3,3,padding='same',name='Conv3d_4_s9_3x3x3')

	branch_1 = conv3d_bn(branch_2,512,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s9_F')

	x = layers.add([branch_0, branch_1])
	
	x = Activation('relu', name='Conv3d_4_s9_F_Relu')(x)

	#Stage 10
	branch_0 = conv3d_bn(x,512,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s10_loop')

	branch_1 = conv3d_bn(x,256,3,1,1,padding='same',name='Conv3d_4_s10_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,256,3,3,3,padding='same',name='Conv3d_4_s10_3x3x3')

	branch_1 = conv3d_bn(branch_2,512,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s10_F')

	x = layers.add([branch_0, branch_1])

	x = Activation('relu', name='Conv3d_4_s10_F_Relu')(x)

	#Stage 11
	branch_0 = conv3d_bn(x,512,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s11_loop')

	branch_1 = conv3d_bn(x,256,3,1,1,padding='same',name='Conv3d_4_s11_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,256,3,3,3,padding='same',name='Conv3d_4_s11_3x3x3')

	branch_1 = conv3d_bn(branch_2,512,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s11_F')

	x = layers.add([branch_0, branch_1])

	x = Activation('relu', name='Conv3d_4_s11_F_Relu')(x)

	#Stage 12
	branch_0 = conv3d_bn(x,512,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s12_loop')

	branch_1 = conv3d_bn(x,256,3,1,1,padding='same',name='Conv3d_4_s12_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,256,3,3,3,padding='same',name='Conv3d_4_s12_3x3x3')

	branch_1 = conv3d_bn(branch_2,512,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_4_s12_F')

	x = layers.add([branch_0, branch_1])
	
	x = Activation('relu', name='Conv3d_4_s12_F_Relu')(x)


	#Downsampling (spatial and temporal)
	x = MaxPooling3D((2,2,2),strides=(2,2,2),padding='same',name='MaxPoolAfterConv4')(x)

	#Conv5 Layer
	#Stage 1
	branch_0 = conv3d_bn(x,1024,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_5_s1_loop')

	branch_1 = conv3d_bn(x,512,3,1,1,padding='same',name='Conv3d_5_s1_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,512,3,3,3,padding='same',name='Conv3d_5_s1_3x3x3')

	branch_1 = conv3d_bn(branch_2,1024,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_5_s1_F')

	x = layers.add([branch_0, branch_1])
	
	x = Activation('relu', name='Conv3d_5_s1_F_Relu')(x)

	#Stage 2
	branch_0 = conv3d_bn(x,1024,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_5_s2_loop')

	branch_1 = conv3d_bn(x,512,3,1,1,padding='same',name='Conv3d_5_s2_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,512,3,3,3,padding='same',name='Conv3d_5_s2_3x3x3')

	branch_1 = conv3d_bn(branch_2,1024,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_5_s2_F')

	x = layers.add([branch_0, branch_1])

	x = Activation('relu', name='Conv3d_5_s2_F_Relu')(x)

	#Stage 3
	branch_0 = conv3d_bn(x,1024,1,1,1,padding='same',use_activation_fn=False,name='Conv3d_5_s3_loop')

	branch_1 = conv3d_bn(x,512,3,1,1,padding='same',name='Conv3d_5_s3_1x1x3_I')

	branch_2 = conv3d_bn(branch_1,512,3,3,3,padding='same',name='Conv3d_5_s3_3x3x3')

	branch_1 = conv3d_bn(branch_2,1024,3,1,1,padding='same',use_activation_fn=False,name='Conv3d_5_s3_F')

	x = layers.add([branch_0, branch_1])

	x = Activation('relu', name='Conv3d_5_s3_F_Relu')(x)

	#Classification block
	x = AveragePooling3D((2,7,7),strides=(1,1,1),padding='valid',name='global_avg_pool')(x)
	
	x = Flatten(name='flatten')(x)

	x = Dense(classes,activation='softmax',name='prediction')(x)

	inputs = img_input
	#Create Model
	model = Model(inputs,x,name='msn_model')
	return model
