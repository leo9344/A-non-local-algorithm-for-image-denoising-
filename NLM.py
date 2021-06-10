# -*- coding:utf-8 -*-
# @Time   : 2021/6/4 10:43 
# @Author : Leo Li
# @Email  : 846016122@qq.com
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import math
from PIL import Image as IMG
class NLM:
    def __init__(self):
        self.img_path = "noisy_image2.jpg"
        self.noise_path = "noisy_image1.jpg"
        #image = mpimg.imread(self.img_path)
        image = IMG.open(self.img_path)

        image2 = IMG.open(self.noise_path)
        # image2 = image2.resize(image.size, IMG.ANTIALIAS)
        #
        # image.paste(image2)
        image = image.convert('L')
        #self.image = image.reshape(image.shape[0],image.shape[1],3)
        self.image = np.array(image) #转灰度图

        #self.image = self.gauss_noise(self.image)

    def gauss_noise(self,image, mean=0, var=0.001):
        image = np.array(image/255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out*255)
        return out

    def GaussianTemplate(self, kernel_size, sigma):
        '''
        Formula H_{i,j}=\frac{1}{2\pi \sigma^2}e^{-\frac{(i-k-1)^2+(j-k-1)^2}{2\sigma^2}}
        H_ij = 1/(2*pi*sigma^2)*exp(-((i-k-1)^2+(j-k-1)^2)/(2*sigma^2))
                k1                                            k2
        kernelsize = 2*k+1
        '''

        template = np.ones(shape = (kernel_size, kernel_size), dtype="float64", )

        k = int(kernel_size / 2)
        k1 = 1/(2*np.pi*sigma*sigma)
        k2 = 2*sigma*sigma

        SUM = 0.0
        for i in range(kernel_size):
            for j in range(kernel_size):
                template[i,j] = k1*np.exp(-((i-k)*(i-k)+(j-k)*(j-k)) / k2)
                SUM += template[i,j]
        # 归一化
        # upleft = 1/template[0,0]
        # for i in range(kernelsize):
        #     for j in range(kernelsize):
        #         template[i,j] *= upleft
        # 小数形式的模板需要除以和
        for i in range(kernel_size):
            for j in range(kernel_size):
                template[i,j] /= SUM
        print("Gaussian Template = \n", template)
        return template

    def Gaussian_Filtering(self, src, dst = [], kernel_size = 3, sigma=0.8):
        print("Gaussian Filtering start. Kernel size = {}, sigma = {}".format(kernel_size,sigma))
        start_time = time.time()
        if kernel_size == 1:
            dst = src
            return dst
        elif kernel_size%2 == 0 or kernel_size <= 0:
            print("卷积核大小必须是大于1的奇数")
            return 0

        padding_size = int((kernel_size - 1) / 2)
        img_width = src.shape[0] + padding_size*2
        img_height = src.shape[1] + padding_size*2

        tmp = np.zeros((img_width,img_height))
        dst = np.zeros((src.shape[0],src.shape[1]))
        #padding
        for i in range(padding_size,img_width-padding_size):
            for j in range(padding_size,img_height-padding_size):
                tmp[i,j] = src[i-padding_size,j-padding_size]
        kernel = self.GaussianTemplate(kernel_size, sigma)
        #Gaussian Filtering
        for row in range(padding_size, img_width-padding_size):
            for col in range(padding_size, img_height-padding_size):
                #sum = [0.0,0.0,0.0] #3Channel
                sum = 0
                for i in range(-padding_size,padding_size+1):
                    for j in range(-padding_size,padding_size+1):
                        #for channel in range(0,3):
                            #sum[channel] += tmp[(row+i),(col+j),channel] * kernel[i+padding_size,j+padding_size]
                        sum += tmp[(row+i),(col+j)] * kernel[i+padding_size,j+padding_size]
                # for channel in range(3):
                #     if sum[channel] > 255:
                #         sum[channel] = 255
                #     if sum[channel] < 0:
                #         sum[channel] = 0
                #     dst[row - padding_size, col - padding_size,channel] = sum[channel]
                if sum > 255:
                    sum = 255
                if sum < 0:
                    sum = 0
                dst[row - padding_size, col - padding_size] = sum
        dst = dst.astype('int32')
        end_time = time.time()

        print('Gaussian Filtering Complete. Time:{}'.format(end_time-start_time))
        method_noise = src - dst

        # plt.subplot(1,3,1)
        # plt.imshow(src,cmap="gray")
        # plt.title("Source Image")
        # plt.subplot(1,3,2)
        # plt.imshow(dst,cmap = "gray")
        # plt.title("Gaussian Filtering")
        # plt.subplot(1, 3, 3)
        # plt.imshow(method_noise, cmap="gray")
        # plt.title("Gaussian Filtering")
        # plt.show()
        return dst, method_noise

    def Anisotropic_Filtering(self, src, dst, iterations = 10, k = 15, _lambda = 0.25):
        print("Anisotropic Filtering start. iterations = {}, k = {}, lambda = {}".format(iterations,k,_lambda))
        start_time = time.time()
        image_width = src.shape[0]
        image_height = src.shape[1]
        k2 = k*k                                  # Since we only need k^2
        old_dst = src.copy()
        new_dst = src.copy()


        for i in range(iterations):
            N_grad,S_grad,W_grad,E_grad = 0,0,0,0 #四邻域梯度
            N_c,S_c,W_c,E_c = 0,0,0,0 #四邻域导热系数
            for row in range(1,image_width-1):
                for col in range(1,image_height-1):
                    # for channel in range(0,3):
                    #     N_grad = int(old_dst[row-1,col,channel]) - int(old_dst[row,col,channel])
                    #     S_grad = int(old_dst[row+1,col,channel]) - int(old_dst[row,col,channel])
                    #     E_grad = int(old_dst[row,col-1,channel]) - int(old_dst[row,col,channel])
                    #     W_grad = int(old_dst[row,col+1,channel]) - int(old_dst[row,col,channel])
                    #     N_c = np.exp(-N_grad*N_grad/k2)
                    #     S_c = np.exp(-S_grad*S_grad/k2)
                    #     E_c = np.exp(-E_grad*E_grad/k2)
                    #     W_c = np.exp(-W_grad*W_grad/k2)
                    #     new_dst[row,col,channel] = old_dst[row,col,channel] + int(_lambda *(N_grad*N_c + S_grad*S_c + E_grad*E_c + W_grad*W_c))
                    N_grad = int(old_dst[row-1,col]) - int(old_dst[row,col])
                    S_grad = int(old_dst[row+1,col]) - int(old_dst[row,col])
                    E_grad = int(old_dst[row,col-1]) - int(old_dst[row,col])
                    W_grad = int(old_dst[row,col+1]) - int(old_dst[row,col])
                    N_c = np.exp(-N_grad*N_grad/k2)
                    S_c = np.exp(-S_grad*S_grad/k2)
                    E_c = np.exp(-E_grad*E_grad/k2)
                    W_c = np.exp(-W_grad*W_grad/k2)
                    new_dst[row,col] = old_dst[row,col] + int(_lambda *(N_grad*N_c + S_grad*S_c + E_grad*E_c + W_grad*W_c))
            old_dst = new_dst

        dst = new_dst
        # plt.subplot(1, 2, 1)
        # plt.imshow(src)
        # plt.title("Source Image")
        # plt.subplot(1, 2, 2)
        # plt.imshow(dst)
        # plt.title("Anisotropic Filtering")
        # plt.show()
        end_time = time.time()
        print("Anisotropic filtering complete. Time:{}".format(end_time-start_time))
        method_noise = src - dst
        return dst,method_noise
    def Total_Variation_Minimization(self,src,dst,iterations = 100,_lambda = 0.03):
        print("Total Variation Minimization start. iterations = {}, lambda = {}".format(iterations,_lambda))
        start_time = time.time()
        image_width = src.shape[0]
        image_height = src.shape[1]
        dst = src.copy()
        u0 = src.copy()
        h = 1
        Energy = []
        cnt= 0
        for i in range(0,iterations):
            for row in range(1,image_width-1):
                for col in range(1,image_height-1):
                    # for channel in range(3):
                    #     ux = (float(dst[row+1,col,channel]) - float(dst[row,col,channel]))/h
                    #     uy = (float(dst[row,col+1,channel]) - float(dst[row,col-1,channel]))/(2*h)
                    #     grad_u = math.sqrt(ux*ux+uy*uy)
                    #     c1 = 0
                    #     if grad_u == 0:
                    #         cnt += 1
                    #     else:
                    #         c1 = 1/grad_u
                    #
                    #     ux = (float(dst[row, col, channel]) - float(dst[row-1, col, channel])) / h
                    #     uy = (float(dst[row-1, col + 1, channel]) - float(dst[row-1, col - 1, channel])) / (2 * h)
                    #     grad_u = math.sqrt(ux * ux + uy * uy)
                    #     c2 = 0
                    #     if grad_u == 0:
                    #         cnt += 1
                    #     else:
                    #         c2 = 1 / grad_u
                    #
                    #     ux = (float(dst[row + 1, col, channel]) - float(dst[row-1, col, channel])) / (2 * h)
                    #     uy = (float(dst[row, col + 1, channel]) - float(dst[row, col, channel])) / h
                    #     grad_u = math.sqrt(ux * ux + uy * uy)
                    #     c3 = 0
                    #     if grad_u == 0:
                    #         cnt += 1
                    #     else:
                    #         c3 = 1 / grad_u
                    #
                    #     ux = (float(dst[row + 1, col-1, channel]) - float(dst[row-1, col-1, channel])) / (2 * h)
                    #     uy = (float(dst[row, col, channel]) - float(dst[row, col - 1, channel])) / h
                    #     grad_u = math.sqrt(ux * ux + uy * uy)
                    #     c4 = 0
                    #     if grad_u == 0:
                    #         cnt += 1
                    #     else:
                    #         c4 = 1 / grad_u
                    #
                    #     dst[row,col,channel] = (u0[row,col,channel] + (1/(_lambda*h*h)) * (c1*dst[row+1,col,channel] + c2*dst[row-1,col,channel] + c3*dst[row,col+1,channel] + c4*dst[row,col-1,channel]) ) * (1/(1+(1/(_lambda*h*h)*(c1+c2+c3+c4))))
                    ux = (float(dst[row + 1, col]) - float(dst[row, col])) / h
                    uy = (float(dst[row, col + 1]) - float(dst[row, col - 1])) / (2 * h)
                    grad_u = math.sqrt(ux * ux + uy * uy)
                    c1 = 0
                    if grad_u == 0:
                        cnt += 1
                    else:
                        c1 = 1 / grad_u

                    ux = (float(dst[row, col]) - float(dst[row - 1, col])) / h
                    uy = (float(dst[row - 1, col + 1]) - float(dst[row - 1, col - 1])) / (2 * h)
                    grad_u = math.sqrt(ux * ux + uy * uy)
                    c2 = 0
                    if grad_u == 0:
                        cnt += 1
                    else:
                        c2 = 1 / grad_u

                    ux = (float(dst[row + 1, col]) - float(dst[row - 1, col])) / (2 * h)
                    uy = (float(dst[row, col + 1]) - float(dst[row, col])) / h
                    grad_u = math.sqrt(ux * ux + uy * uy)
                    c3 = 0
                    if grad_u == 0:
                        cnt += 1
                    else:
                        c3 = 1 / grad_u

                    ux = (float(dst[row + 1, col - 1]) - float(dst[row - 1, col - 1])) / (2 * h)
                    uy = (float(dst[row, col]) - float(dst[row, col - 1])) / h
                    grad_u = math.sqrt(ux * ux + uy * uy)
                    c4 = 0
                    if grad_u == 0:
                        cnt += 1
                    else:
                        c4 = 1 / grad_u

                    dst[row, col] = (u0[row, col] + (1 / (_lambda * h * h)) * (
                                c1 * dst[row + 1, col] + c2 * dst[row - 1, col] + c3 * dst[
                            row, col + 1] + c4 * dst[row, col - 1])) * (
                                                         1 / (1 + (1 / (_lambda * h * h) * (c1 + c2 + c3 + c4))))
            # 处理边缘
            for row in range(1,image_width-1):
                dst[row,0] = dst[row,1]
                dst[row,image_height-1] = dst[row,image_height-1-1]
            for col in range(1,image_height-1):
                dst[0,col] = dst[1,col]
                dst[image_width-1,col] = dst[image_width-1-1,col]

            dst[0,0] = dst[1,1]
            dst[0,image_height-1] = dst[1,image_height-1-1]
            dst[image_width-1,0] = dst[image_width-1-1,1]
            dst[image_width-1,image_height-1] = dst[image_width-1-1,image_height-1-1]

            energy = 0
            for row in range(1, image_width - 1):
                for col in range(1, image_height - 1):
                    # for channel in range(3):
                    #     ux = (float(dst[row+1,col,channel]) - float(dst[row,col,channel]))/h
                    #     uy = (float(dst[row,col+1,channel]) - float(dst[row,col,channel]))/h
                    #     tmp = (float(u0[row,col,channel]) - float(dst[row,col,channel]))
                    #     fid = tmp*tmp
                    #     energy += math.sqrt(ux*ux + uy*uy) + _lambda*fid
                    ux = (float(dst[row+1,col]) - float(dst[row,col]))/h
                    uy = (float(dst[row,col+1]) - float(dst[row,col]))/h
                    tmp = (float(u0[row,col]) - float(dst[row,col]))
                    fid = tmp*tmp
                    energy += math.sqrt(ux*ux + uy*uy) + _lambda*fid
            Energy.append(energy)
        end_time = time.time()
        print('Total Variation Minimization Complete. Time:{}'.format((end_time - start_time)))
        method_noise = src - dst
        # plt.imshow(dst,cmap = "gray")
        # plt.show()
        return dst,method_noise

    def Yaroslavsky_Filtering(self,src,dst,kernel_size=3,h=1):
        print("Yaroslavsky Filtering start. Kernel size = {}, h = {}".format(kernel_size, h))
        start_time = time.time()
        image_width = src.shape[0]
        image_height = src.shape[1]
        weight = np.zeros(src.shape)
        dst = src.copy()
        padding_size = int((kernel_size - 1) / 2)
        for row in range(padding_size, image_width - padding_size):
            for col in range(padding_size, image_height - padding_size):
                sum = 0
                for i in range(-padding_size, padding_size + 1):
                    for j in range(-padding_size, padding_size + 1):
                        if i == 0 and j == 0:
                            continue
                        sum += np.exp(-(int(src[(row + i), (col + j)]) - int(src[row,col]))**2/(h*h))
                weight[row,col] = sum

        for row in range(padding_size, image_width - padding_size):
            for col in range(padding_size, image_height - padding_size):
                sum = 0
                sum_weight = 0
                for i in range(-padding_size, padding_size + 1):
                    for j in range(-padding_size, padding_size + 1):
                        sum += weight[(row+i),(col+j)]*int(src[(row+i),(col+j)])
                        sum_weight += weight[(row+i),(col+j)]
                dst[row,col] = sum/sum_weight
        end_time = time.time()
        print('Yaroslavsky Filtering Complete. Time:{}'.format(end_time - start_time))
        method_noise = src - dst
        # plt.imshow(dst,cmap = "gray")
        # plt.show()
        return dst, method_noise


    def Max_Array(self,array1,array2,width,height):
        result = np.zeros((width,height))
        for row in range(0,width):
            for col in range(0,height):
                result[row,col] = max(array1[row,col],array2[row,col])
        return result

    def integralImgSqDiff2(self,padded_img,Ds,search_x,search_y):
        width,height = padded_img.shape[0],padded_img.shape[1]
        Dist2 = (padded_img[Ds:-Ds , Ds:-Ds] -
                 padded_img[Ds + search_x: width - Ds + search_x,
                            Ds + search_y: height - Ds + search_y]) **2
        Sd = Dist2.cumsum(0)
        Sd = Sd.cumsum(1)
        # Weighted Euclidean distance = Sd
        return Sd

    def NL_Means(self,src,dst,neighborhood_size = 7,search_window_size = 21,h = 10):
        """
        :param src: sourse image
        :param dst: dst image
        :param neighborhood_size: 邻域窗口大小
        :param search_window_size: 搜索窗口大小
        :param h:  高斯函数平滑参数
        :return:   去噪后图像
        """
        print("NL means start. neighborhood size = {}, search window size = {}, h = {}".format(neighborhood_size,search_window_size,h))
        start_time = time.time()
        dst = np.zeros(src.shape)
        ds = int(neighborhood_size/2)
        Ds = int(search_window_size/2)

        img_width,img_height = src.shape[0],src.shape[1]
        length0, length1 = img_width + 2 * Ds, img_height + 2 * Ds
        padded_img = np.pad(src, ds+Ds+1 ,'symmetric').astype('float64')
        padded_v = np.pad(src,ds+Ds+1,'symmetric').astype("float64")
        avg = np.zeros(src.shape)
        weight = np.zeros(src.shape)
        wmax = np.zeros(src.shape)
        h2 = h * h
        d = neighborhood_size ** 2

        for search_x in range(-Ds,Ds+1):
            for search_y in range(-Ds,Ds+1):
                if search_x == 0 and search_y == 0 :
                    continue
                Sd = self.integralImgSqDiff2(padded_img,Ds,search_x,search_y)
                SqDist2 = Sd[2 * ds + 1:-1, 2 * ds + 1:-1] + Sd[0:-2 * ds - 2, 0:-2 * ds - 2] - \
                          Sd[2 * ds + 1:-1, 0:-2 * ds - 2] - Sd[0:-2 * ds - 2, 2 * ds + 1:-1]
                SqDist2 /= (d*h2)
                w = np.exp(-SqDist2)
                v = padded_v[Ds + search_x:length0 - Ds + search_x, Ds + search_y:length1 - Ds + search_y]
                avg += w*v
                wmax = self.Max_Array(wmax,w,img_width,img_height)
                weight += w

        avg += wmax * src
        avg /= wmax + weight
        dst = avg.astype('uint8')

        end_time = time.time()
        print('NL means Complete. Time:{}'.format(end_time - start_time))
        method_noise = src - dst
        # plt.subplot(1,3,1)
        # plt.imshow(src,cmap="gray")
        # plt.subplot(1,3,2)
        # plt.imshow(dst,cmap="gray")
        # plt.subplot(1,3,3)
        # plt.imshow(method_noise,cmap="gray")
        # plt.show()
        return dst, method_noise

    def double2uint8(self,I, ratio=1.0):
        return np.clip(np.round(I * ratio), 0, 255).astype(np.uint8)

    def make_kernel(self,f):
        kernel = np.zeros((2 * f + 1, 2 * f + 1))
        for d in range(1, f + 1):
            kernel[f - d:f + d + 1, f - d:f + d + 1] += (1.0 / ((2 * d + 1) ** 2))
        return kernel / kernel.sum()

    def NLmeansfilter(self,I, h_=10, templateWindowSize=5, searchWindowSize=11):
        f = templateWindowSize / 2
        t = searchWindowSize / 2
        height, width = I.shape[:2]
        padLength = t + f
        I2 = np.pad(I, padLength, 'symmetric')
        kernel = self.make_kernel(f)
        h = (h_ ** 2)
        I_ = I2[padLength - f:padLength + f + height, padLength - f:padLength + f + width]

        average = np.zeros(I.shape)
        sweight = np.zeros(I.shape)
        wmax = np.zeros(I.shape)
        for i in range(-t, t + 1):
            for j in range(-t, t + 1):
                if i == 0 and j == 0:
                    continue
                I2_ = I2[padLength + i - f:padLength + i + f + height, padLength + j - f:padLength + j + f + width]
                w = np.exp(-cv2.filter2D((I2_ - I_) ** 2, -1, kernel) / h)[f:f + height, f:f + width]
                sweight += w
                wmax = np.maximum(wmax, w)
                average += (w * I2_[f:f + height, f:f + width])
        return (average + wmax * I) / (sweight + wmax)


NLM = NLM()
results = []
method_noise = []
names = ['Source Image', 'Gaussian Filtering', 'Anisotropic Filtering','Total Variation Minimization','Yaroslavsky Filtering', 'NL-Means']
dst = []
dst1,method_noise1 = NLM.Gaussian_Filtering(NLM.image, dst, 5, 2)
dst2,method_noise2 = NLM.Anisotropic_Filtering(NLM.image, dst, 20, 20, 0.25)
dst3,method_noise3 = NLM.Total_Variation_Minimization(NLM.image,dst,100,0.03)
dst4,method_noise4 = NLM.Yaroslavsky_Filtering(NLM.image,dst,4,5)
dst5,method_noise5 = NLM.NL_Means(NLM.image,dst,7,21,2.5)

results.append(NLM.image)
results.append(dst1)
results.append(dst2)
results.append(dst3)
results.append(dst4)
results.append(dst5)

method_noise.append(NLM.image-NLM.image)
method_noise.append(method_noise1)
method_noise.append(method_noise2)
method_noise.append(method_noise3)
method_noise.append(method_noise4)
method_noise.append(method_noise5)

plt.figure(figsize=(16,9))
for i in range(0,len(results)):
    plt.subplot(2,len(results),i+1)
    plt.imshow(results[i],cmap = "gray")
    plt.title(names[i])
for i in range(0,len(results)):
    plt.subplot(2, len(results), len(results) + i + 1)
    plt.imshow(method_noise[i], cmap="gray")
    plt.title(names[i])
plt.show()

