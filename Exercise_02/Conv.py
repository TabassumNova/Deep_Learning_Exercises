import numpy as np
from scipy import ndimage
from scipy import signal
from .Base import BaseLayer
import math
from scipy.ndimage import correlate1d
class Conv(BaseLayer):
    def __init__(self,  stride_shape, convolution_shape, num_kernels):
        BaseLayer.__init__(self)
        self.trainable = True

        if len(stride_shape) == 1:
            self.stride_shape_y, self.stride_shape_x = stride_shape[0], stride_shape[0]
            self.num_kernels = num_kernels
        elif len(stride_shape) == 2:
            self.stride_shape_y, self.stride_shape_x = stride_shape[0], stride_shape[1]

        self.num_kernels = num_kernels
        self.convolution_shape = convolution_shape
        if len(convolution_shape) == 2:
            self.conv_c = convolution_shape[0]
            self.conv_m = convolution_shape[1]
            self.conv_n = 1
            self._weights = np.empty([num_kernels, self.conv_c, self.conv_m])
            for i in range (0, self.num_kernels):
                self._weights[i] = np.random.rand(self.conv_c, self.conv_m)
        elif len(convolution_shape) == 3:
            self.conv_c = convolution_shape[0]
            self.conv_m = convolution_shape[1]
            self.conv_n = convolution_shape[2]
            self._weights = np.empty([num_kernels, self.conv_c, self.conv_m, self.conv_n])
            for i in range (0, self.num_kernels):
                self._weights[i] = np.random.rand(self.conv_c, self.conv_m, self.conv_n)
        bias = []
        for i in range(0, self.num_kernels):
            bias.append(float(np.random.rand(1)))
        self._bias = np.array(bias)
        #self._bias = None
        self._optimizer = None
        self._gradient_weights = None
        self._gradient_bias = None
        self.image_status = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias):
        self._bias = bias

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if len(input_tensor.shape) == 3:
            self.image_status = "1D"
        elif len(input_tensor.shape) == 4:
            self.image_status = "2D"
        batch_size = input_tensor.shape[0] #(b,c,y) or (b,c,y,x)
        next_input_tensor = []
        for img in range(0, batch_size):
            input = input_tensor[img] #(c,y) or (c,y,x)
            all_kernel = []
            for kernel in range(0, self.num_kernels):
                n = signal.correlate(input, self._weights[kernel], mode='same')
                if self.image_status == "1D":
                    if self.conv_c <= 2:
                        next = np.add(n[self.conv_c-1], self._bias[kernel]).reshape(1,-1)
                    else:
                        next = np.add(n[-2], self._bias[kernel]).reshape(1,-1)
                if self.image_status == "2D":
                    if self.conv_c <= 2:
                        next = np.add(n[self.conv_c-1], self._bias[kernel])
                    else:
                        next = np.add(n[-2], self._bias[kernel])
                # Sampling
                sampled = []
                for row in range(0, math.ceil(next.shape[0]/self.stride_shape_y)):
                    coloumn_list = []
                    for coloumn in range(0, math.ceil(next.shape[1]/self.stride_shape_x)):
                        coloumn_list.append(next[row * self.stride_shape_y, coloumn * self.stride_shape_x])
                    coloumn_array = np.array(coloumn_list)
                    sampled.append(coloumn_array)  #(y,x)
                if self.image_status == "1D":
                    sampled_array = np.array(sampled).squeeze(axis=0)
                else:
                    sampled_array = np.array(sampled)
                all_kernel.append(sampled_array)   #(k,y,x)
            all_kernel_array = np.array(all_kernel)
            next_input_tensor.append(all_kernel_array) #(b,k,y,x)
        next_input_tensor_array = np.array(next_input_tensor)
        #print("forward input: ",self.input_tensor.shape)
        #print("forward output: ",next_input_tensor_array.shape)

        return next_input_tensor_array

    def backward(self, error_tensor):
        #print("backward input: ",error_tensor.shape)
        batch_size = error_tensor.shape[0]  # (b,k,y,x)
        error_batch = []
        gradient_all_img = []
        ## Weight_common
        weights_all = []
        for channel in range(0, self.conv_c):
            w = []
            for kernel in range(0, self.num_kernels):    #for ch1 : k1ch1, k2ch1... for ch2: k1ch2, k2ch2...
                w.append(np.flip(self._weights[kernel, channel]))
                #w.append((self._weights[kernel, channel]))
            weight_per_channel = np.array(w)
            weights_all.append(weight_per_channel)
        weights_backward = np.array(weights_all)

        ## Calculation of En-1
        for img in range(0, batch_size):
            error = error_tensor[img]  # (k,y,x)
            error_prev = []
            e1 = []

            for channel in range(0, self.conv_c):
                e = ndimage.convolve(error, weights_backward[channel], mode='constant', cval=0.0) #e(k,y,x) * weight(k,y,x) = (k,y,x)
                #scipy.signal.convolve(mode = same)
                if self.num_kernels <= 2:
                    e1.append(e[self.num_kernels-2])
                else:
                    e1.append(e[-2])
                """
                if self.conv_c <= 2:
                    e1.append(e[self.conv_c-1])
                else:
                    e1.append(e[-2])
                """
                if self.stride_shape_y == 1 and self.stride_shape_x == 1:
                    error_prev = e1
                    pass
                elif self.stride_shape_y != 1 or self.stride_shape_x != 1:
                    if self.image_status == "1D":
                        upsampled = []
                        for i in range(0, e1[channel].shape[0]):
                            upsampled.append(e1[channel][i])
                            if len(upsampled) < self.input_tensor.shape[-1]:
                                for j in range(0, self.stride_shape_y-1):
                                    upsampled.append(0)
                                    # new
                                    if len(upsampled) == self.input_tensor.shape[-1]:
                                        break
                                    # new
                    elif self.image_status == "2D":
                        upsampled = []
                        upsampled_row = []
                        if self.stride_shape_x != 1:
                            for row in range(0, e1[channel].shape[0]):
                                upsampled_coloumn = []
                                for coloumn in range(0, e1[channel].shape[1]):
                                    upsampled_coloumn.append(e1[channel][row, coloumn])
                                    if len(upsampled_coloumn) < self.input_tensor.shape[-1]:
                                        for j in range(0, self.stride_shape_x-1):
                                            upsampled_coloumn.append(0)
                                            # new
                                            if len(upsampled_coloumn) == self.input_tensor.shape[-1]:
                                                break
                                            # new
                                upsampled_coloumn_array = np.array(upsampled_coloumn)
                                upsampled_row.append(upsampled_coloumn_array)
                        else:
                            upsampled_row.append(e1[channel])

                        if self.stride_shape_y != 1:
                            for row in range(0, e1[channel].shape[0]):
                                upsampled.append(upsampled_row[row])
                                if len(upsampled) < self.input_tensor.shape[-2]:
                                    for i in range(0, self.stride_shape_y-1):
                                        upsampled.append(np.zeros((upsampled_row[row].shape)))
                                        # new
                                        if len(upsampled) == self.input_tensor.shape[-2]:
                                            break
                                        # new
                        else:
                            upsampled = upsampled_row
                    error_prev.append(upsampled)
            error_prev_array = np.array(error_prev)
            error_batch.append(error_prev_array)
        error_batch_array = np.array(error_batch)
        #print("backward output: ",error_batch_array.shape)

        ## calculation of gradient weights
        added_gradient = np.zeros((self.weights.shape))
        added_bias = np.zeros((self.bias.shape))
        for img in range(0, batch_size):
            error = error_tensor[img] #(c,y,x)
            # Gradient calculation
            combined_kernel = []
            bias = []
            for kernel in range(0, self.num_kernels):
                combined_channel = []
                for channel in range(0, self.conv_c):
                    if self.image_status == "1D":
                        #padded_input = np.pad(self.input_tensor[img, channel], ((pad, pad)), mode='constant', constant_values=(0, 0))
                        in_tensor = self.input_tensor[img, channel]
                        if self.stride_shape_y == 1:
                            out = correlate1d(in_tensor, weights=error[kernel])
                        else:
                            upsampled_coloumn = []
                            count = 1
                            for col in error[kernel]:
                                upsampled_coloumn.append(col)
                                if count == error[kernel].shape[0]:
                                    break
                                else:
                                    for j in range(0, self.stride_shape_y - 1):
                                        upsampled_coloumn.append(0)
                                count = count+1
                            w_array = np.array(upsampled_coloumn)
                            out = signal.correlate2d(in_tensor.reshape(-1,1), w_array.reshape(-1,1), mode='valid')[:,0]
                    elif self.image_status == "2D":
                        pad_m = math.floor(self.conv_m / 2)
                        pad_n = math.floor(self.conv_n / 2)
                        padded_input = np.pad(self.input_tensor[img, channel], ((pad_m, pad_m), (pad_n, pad_n)),
                                              mode='constant', constant_values=(0, 0))
                        ## new block
                        err_kernel = []
                        if self.stride_shape_y == 1 and self.stride_shape_x == 1:
                            err = error[kernel]
                            #pass
                        elif self.stride_shape_y != 1 or self.stride_shape_x != 1:
                            upsampled = []
                            upsampled_row = []
                            if self.stride_shape_x != 1:
                                for row in range(0, error[kernel].shape[0]):
                                    upsampled_coloumn = []
                                    # count = 1
                                    # for col in error[kernel,row]:
                                    #     upsampled_coloumn.append(col)
                                    #     if count == error[kernel, row].shape[0]:
                                    #         break
                                    #     else:
                                    #         for j in range(0, self.stride_shape_x - 1):
                                    #             upsampled_coloumn.append(0)
                                    #     count = count + 1
                                    for coloumn in range(0, error[kernel].shape[1]):
                                        #new loop
                                        # c = 0
                                        # cl = error[kernel][row][c]
                                        # while cl in error[kernel][row]:
                                        #     c = c+1
                                        #     upsampled_coloumn.append(cl)
                                        #     for j in range(0, self.stride_shape_x - 1):
                                        #         upsampled_coloumn.append(0)
                                        #     if c > error[kernel][row].shape[0]:
                                        #         break
                                        #     else:
                                        #         cl = error[kernel][row][c]
                                        #new loop
                                        upsampled_coloumn.append(error[kernel][row, coloumn])
                                        if len(upsampled_coloumn) < self.input_tensor.shape[-1]:
                                            for j in range(0, self.stride_shape_x - 1):
                                                upsampled_coloumn.append(0)
                                                # new
                                                if len(upsampled_coloumn) == self.input_tensor.shape[-1]:
                                                    break
                                                # new
                                    # test = upsampled_coloumn[:self.weights[0,0].shape[1]]
                                    # upsampled_coloumn_array = np.array(test)
                                    upsampled_coloumn_array = np.array(upsampled_coloumn)
                                    upsampled_row.append(upsampled_coloumn_array)
                            else:
                                upsampled_row.append(error[kernel])
                            if self.stride_shape_y != 1:
                                for row in range(0, error[kernel].shape[0]):
                                    upsampled.append(upsampled_row[row])
                                    if len(upsampled) < self.input_tensor.shape[-2]:
                                        for i in range(0, self.stride_shape_y - 1):
                                            upsampled.append(np.zeros((upsampled_row[row].shape)))
                                            # new
                                            if len(upsampled) == self.input_tensor.shape[-2]:
                                                break
                                            # new
                            else:
                                upsampled = upsampled_row
                            err_kernel.append(upsampled)
                            err = np.array(upsampled)
                        #print("padded input: ", padded_input.shape)
                        #print("err: ", err.shape)
                        out_pre = signal.correlate2d(padded_input, err, mode='valid')  # input_tensor(y,x) conv error(y,x)
                        ##new block
                        #out = signal.correlate2d(padded_input, error[kernel], mode='valid')  #input_tensor(y,x) conv error(y,x)
                        #print("out.shape: ",out.shape)
                        #print("padded input: ", padded_input.shape)
                        out = out_pre[:self.weights.shape[2], :self.weights.shape[3]]
                    combined_channel.append(out)
                combined_channel_array = np.array(combined_channel)
                combined_kernel.append(combined_channel_array)
                #for bias
                bias.append(np.sum(error[kernel]))
            bias_array = np.array(bias)
            added_bias = np.add(added_bias, bias_array)
            combined_kernel_array_img = np.array(combined_kernel)
            added_gradient = np.add(added_gradient, combined_kernel_array_img)
        self._gradient_weights = added_gradient
        self._gradient_bias = added_bias
        #self._gradient_weights = combined_kernel_array_img
            #gradient_all_img.append(combined_kernel_array_img)
        #gradient_all_img_array = np.array(gradient_all_img)
        #print("Gradient: ",added_gradient.shape)
        #print(gradient_all_img_array.shape)
        #print("weight.shape: ", self.weights.shape)
        if self._optimizer != None:
            self._weights = self._optimizer.calculate_update(self._weights, self._gradient_weights)
            self._bias = self._optimizer.calculate_update(self._bias, self._gradient_bias)
        return error_batch_array

    def initialize(self, weights_initializer, bias_initializer):
        weights_shape = self._weights.shape
        bias_shape = self._bias.shape
        fan_in = self.conv_c * self.conv_m * self.conv_n
        fan_out = self.num_kernels * self.conv_m * self.conv_n
        self._weights = weights_initializer.initialize(weights_shape, fan_in, fan_out)
        self._bias = bias_initializer.initialize(bias_shape, fan_in, fan_out)


