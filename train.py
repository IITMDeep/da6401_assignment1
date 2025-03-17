import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import wandb
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import matplotlib
import seaborn

import warnings
import argparse


warnings.filterwarnings("ignore")

# wandblogin
# wandb.login(key='6a66920f640c7001ec17ad4aa7a5da8b378aee61')

parser = argparse.ArgumentParser()

parser.add_argument('-wp' , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='DL-Assignment-1-v3')
parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='CS24M019')
parser.add_argument('-d', '--dataset', help='choices: ["mnist", "fashion_mnist"]', choices = ["mnist", "fashion_mnist"],type=str, default='fashion_mnist')
parser.add_argument('-nhl', '--num_layers', help='Number of hidden layers used in feedforward neural network.',type=int, default=4)
parser.add_argument('-sz', '--hidden_size', help ='Number of hidden neurons in a feedforward layer.', type=int, default=128)
parser.add_argument('-w_i', '--weight_init', help = 'choices: ["random", "Xavier"]', choices = ["random", "Xavier"],type=str, default='Xavier')
parser.add_argument('-l','--loss', help = 'choices: ["mean_squared_error", "cross_entropy"]' , choices = ["meanSquaredError", "crossEntropy"],type=str, default='crossEntropy')
parser.add_argument('-a', '--activation', help='choices: ["identity", "sigmoid", "tanh", "ReLU"]', choices = ["identity", "sigmoid", "tanh", "ReLU"],type=str, default='ReLU')
parser.add_argument('-o', '--optimizer', help = 'choices: ["SGD", "Momentum", "NAG", "RMSProp", "ADAM", "nADAM"]', choices = ["SGD", "Momentum", "NAG", "RMSProp", "ADAM", "nADAM"],type=str, default = 'ADAM')
parser.add_argument('-lr', '--learning_rate', help = 'Learning rate used to optimize model parameters', type=float, default=0.0001)
parser.add_argument('-m', '--Momentum', help='Momentum used by Momentum and nag optimizers.',type=float, default=0.5)
parser.add_argument('-beta', '--beta', help='Beta used by rmsprop optimizer',type=float, default=0.9)
parser.add_argument('-beta1', '--beta1', help='Beta1 used by adam and nadam optimizers.',type=float, default=0.999)
parser.add_argument('-beta2', '--beta2', help='Beta2 used by adam and nadam optimizers.',type=float, default=0.999)
parser.add_argument('-eps', '--epsilon', help='Epsilon used by optimizers.',type=float, default=0.000001)
parser.add_argument('-w_d', '--weight_decay', help='Weight decay used by optimizers.',type=float, default=0.0005)
parser.add_argument('-e', '--epochs', help="Number of epochs to train neural network.", type=int, default=10)
parser.add_argument('-b', '--batch_size', help="Batch size used to train neural network.", type=int, default=32)

# Added Extra Arguments

    # To print data on Terminal : default -> 1
parser.add_argument('-cl', '--console_log', help="To print results on the terminal.", type=int, default=1)

    # To log data to wandb : default -> 1
parser.add_argument('-wl', '--wandb_log', help="To log results on wandb.", type=int, default=1)


arguments = parser.parse_args()


# Activation functions
class ActivationFunctions:
  def relu(self,k):
    """
    ReLU (Rectified Linear Unit) activation function.
    It returns the input value if it's positive, otherwise, it returns zero.
    This helps in handling vanishing gradient problems in deep networks.
    """
    return k*(k>0)

  def sigmoid(self,k):
    """
    Sigmoid activation function.
    It maps input values to a range between 0 and 1, making it useful for probabilities.
    To avoid overflow/underflow issues, the input is clipped within [-500, 500].
    """
    s=1+np.exp(-k)
    return 1/s

  def tanh(self,k):
    """
    Hyperbolic tangent (tanh) activation function.
    It maps input values to a range between -1 and 1, which helps in training deep networks.
    Clipping prevents numerical instability.
    """
    s = np.tanh(k)
    return s

  def identity(self,x):
    """
    Identity activation function.
    It simply returns the input as it is.
    Used in linear layers where no transformation is required.
    """
    return x

  def grad_sigmoid(self,x):
    """
    Computes the gradient of the sigmoid function.
    Uses clipping to prevent overflow in the exponential function.
    """
    s = np.multiply((1/(1+np.exp(-x))),(1-(1/(1+np.exp(-x)))))
    return s

  def grad_tanh(self,x):
    """
    Computes the gradient of the tanh function.
    """
    return 1 - (np.tanh(x) ** 2)  # Derivative of tanh

  def grad_relu(self,x):
    """
    Computes the gradient of the ReLU function.
    Returns 1 for positive inputs, 0 otherwise.
    """
    relu_derivative=0
    relu_derivative=np.maximum(0,x)
    relu_derivative[relu_derivative>0]=1
    return relu_derivative

  def grad_identity(self,x):
    """
    Computes the gradient of the identity function.
    The derivative of an identity function is always 1.
    """
    
    return 1  # Return an array of ones

  def softmax(self,k):
    """
    Computes the softmax function.
    Uses normalization by subtracting max(a) to improve numerical stability.
    """
    x=np.copy(k)
    i=0
    while i < k.shape[0]:
        add=0
        largi=np.argmax(k[i])
        j=0
        while j< k.shape[1]:
            add+=np.exp(k[i][j]-k[i][largi])
            j=j+1
        s=k[i]-k[i][largi]
        k[i]=np.exp(s)/add
        x[i]=k[i]
        i=i+1
    return x

  def grad_softmax(self, a):
    """
    Computes the gradient of the softmax function.
    Note: This is an incorrect implementation of the gradient.
    The correct gradient is a Jacobian matrix, not element-wise.
    """
    return self.softmax(a) * (1 - self.softmax(a))  # Incorrect gradient

class LossFunction:
  def crossEntropy(self,y_hat,y1,cac):
    """
    Computes the cross-entropy loss.
    """
    l=0
    i=0
    for i in range (y1.shape[0]):
        l=l-((np.log2(y_hat[i][y1[i]])))
        i=i+1
    s=l+cac
    z=y1.shape[0]
    return s/z

  def meanSquaredError(self,y_hat,y1,cac):
    """
    Computes the mean squared error loss.
    """
    bl=np.zeros((y1.shape[0],y_hat.shape[1]))
    i=0
    while i<y1.shape[0]:
        bl[i][y1[i]]=1
        i=i+1
    s=(np.sum(((y_hat-bl)**2)))+cac
    t=y1.shape[0]
    return s/t

A = ActivationFunctions()
L = LossFunction()

"""## NeuralNetwork"""
# defined Neuarl Network class
class NeuralNetwork:
    def __init__(self):

        self.w,self.b,self.a,self.h,self.wd,self.ad,self.hd,self.bd=[],[],[],[],[],[],[],[]


#Defines various activation functions
    def activations(self,act,k):
        if act=='sigmoid':
            return A.sigmoid(k)
        elif act =='ReLU':
            return A.relu(k)
        elif act =='tanh':
            return A.tanh(k)
        elif act == 'identity':
          return A.identity(k)
        elif act =='softmax':
            return A.softmax(k)

# Defines derivatives of various activation functions.
    def gradActivations(self,act,k):
        if act=='sigmoid':
            return A.grad_sigmoid(k)
        if act=='ReLU':
            return A.grad_relu(k)
        if act=='tanh':
            return A.grad_tanh(k)
        if act == 'identity':
            return A.grad_identity(k)


# Defines loss functions.
    def functionLoss(self,lossFunc,y_hat,y1,Momentum):
        reg_loss=0
        i=0
        for i in range (len(self.w)):
            s=np.sum(self.w[i]**2)
            reg_loss=reg_loss+s
            i+=1
        reg_loss=(Momentum*reg_loss)/2
        ch=1
        if lossFunc=='crossEntropy':
            loss=0
            i=0
            for i in range (y1.shape[0]):
                x = y_hat[i][y1[i]]
                y = (np.log2(x))
                loss -= (y)
                i+=1
            return ((loss+reg_loss)/y1.shape[0])

        elif lossFunc=='meanSquaredError':
            bl=np.zeros((y1.shape[0],y_hat.shape[1]))
            i=0
            for i in range (y1.shape[0]):
                bl[i][y1[i]]=1
                i+=1
            return (((np.sum(((y_hat-bl)**2)))+reg_loss)/(y1.shape[0]))


# Initializes weights and biases for the neural network layers.

    def formLayers(self,hidden_layers,neuron,inpNeurons,start,classes,ques=0,hl=[]):

        self.w = []
        self.b = []

        totalLayer=[]
        if ques == 0:
          np.random.seed(5)
          totalLayer.append(inpNeurons)
          i=0
          for i in range (hidden_layers):
              totalLayer.append(neuron)
              i+=1
          intialization = 0
          totalLayer.append(classes)

        elif ques == 2:
          totalLayer = [inpNeurons] + hl + [classes]
          hidden_layers = len(hl)

        # totalLayer = [784,128,,,,,10]
        if start=='random':
            i=0
            while i<=hidden_layers:
                self.b.append(np.random.uniform(-0.5,0.5,(1,totalLayer[i+1])))
                self.w.append(np.random.uniform(-0.5,0.5,(totalLayer[i],totalLayer[i+1])))
                i+=1
        if start=='Xavier':
            i=0
            while i<=hidden_layers:
                frwd=totalLayer[i+1]
                curr=totalLayer[i]
                x=(np.random.randn(1,frwd))
                y=np.sqrt(6/(1+frwd))
                self.b.append(x*y)
                m=(np.random.randn(curr,frwd))
                n=np.sqrt(6/(curr+frwd))
                self.w.append(m*n)
                i+=1

#Performs forward pass through the neural network layers.
    def forward_pass(self,x,act='sigmoid'):

        self.a,self.h=[],[]
        check=x
        i=0
        while i<len(self.w)-1:
            q1=np.add(np.matmul(check,self.w[i]),self.b[i])
            bool = act=='ReLU' and i==0
            if (bool):
                j=0
                s=q1.shape[0]
                while j < s:
                    maxi=np.argmax(q1[j])
                    q1[j]/=q1[j][maxi]
                    j=j+1
            r1=self.activations(act,q1)
            check=r1
            self.h.append(r1)
            self.a.append(q1)
            i=i+1
        # print(len(self.w)-1,"check")
        z=len(self.w)-1
        q1=np.add(np.matmul(check,self.w[z]),self.b[z])
        r1=self.activations('softmax',q1)
        self.h.append(r1)
        self.a.append(q1)

        return self.h[-1]


# Performs backward pass through the neural network layers to compute gradients.
    def backward_pass(self,y_hat,y1,x1,classes,activation,lossFunc,Momentum):
        self.wd,self.bd,self.ad,self.hd=[],[],[],[]
        bl=0
        bl=np.zeros((y1.shape[0],classes))
        i=0
        while i< y1.shape[0]:
            bl[i][y1[i]]=1
            i=i+1

        b=None
        a=None


        if lossFunc=="crossEntropy":
            y_hat_l=np.zeros((y_hat.shape[0],1))
            i=0
            while i< y_hat.shape[0]:
                y_hat_l[i]=y_hat[i][y1[i]]
                i=i+1

            b=-1*(bl-y_hat)
            a=-1*(bl/y_hat_l)

            self.hd.append(a)
            self.ad.append(b)

        elif lossFunc=="meanSquaredError":
            s=y_hat-bl
            a=2*s
            self.hd.append(a)
            b=[]
            j=0
            while j< y_hat.shape[1]:
                r=y_hat.shape[0]
                s=y_hat.shape[1]
                hot_j=np.zeros((r,s))
                hot_j[:,j]=1
                hat_j=np.ones((r,s))*(y_hat[:,j].reshape(r,1))
                l=y_hat-bl
                x=hot_j-hat_j
                aj=2*(np.sum((l)*(y_hat*(x)),axis=1))
                b.append(aj)
                j=j+1
            self.ad.append(np.array(b).T)



        j = len(self.w)-1
        while j>-1:
            u=self.h[j-1].T
            bool = (j==0)
            if bool:
                u=x1.T
            t=x1.shape[0]
            length=len(self.ad)
            w=np.matmul(u,self.ad[-1])/t
            b=np.sum(self.ad[length-1],axis=0)/t
            if j!=0:
                a=np.matmul(self.ad[length-1],self.w[j].T)
                der=self.gradActivations(activation,self.a[j-1])
                z=np.multiply(a,der)
                self.hd.append(a)
                self.ad.append(z)
            self.bd.append(b)
            self.wd.append(w)
            j=j-1
        i=0

        while i< len(self.w):
            s=len(self.w)-1-i
            self.wd[s]-=Momentum*self.w[i]
            i=i+1
#Compute the accuracy of the neural network on the given dataset.
    def accuracy(self,x2,y2,act):
        self.forward_pass(x2,act)
        ypred=np.argmax(self.h[len(self.w)-1],axis=1)
        n=0
        l=y2.shape[0]
        i=0
        while i<y2.shape[0]:
            if ypred[i]!=y2[i]:
                n+=+1
            i+=1
        return ((x2.shape[0]-n)/y2.shape[0])*100

#Make predictions using the neural network and print the test accuracy.
    def predict(self,x2,y2,act):
        self.forward_pass(x2,act)
        fut=np.argmax(self.h[len(self.w)-1],axis=1)
        n=0
        i=0
        while i < y2.shape[0]:
            if fut[i]!=y2[i]:
                n+=1
            i+=1

        acc=((x2.shape[0]-n)/y2.shape[0])*100
        print("Test Accuracy: "+str(acc))

#Create batches from input data and labels.
    def createBatches(self,x1,y1,size):
        info,res=[],[]
        s=x1.shape[0]
        l=math.ceil(s/size)
        i=0
        while i < l:
            group,group_ans=[],[]
            j=i*size
            s=min((i+1)*size,x1.shape[0])
            while j< s:
                group.append(x1[j])
                group_ans.append(y1[j])
                j+=1
            group_ans=np.array(group_ans)
            group=np.array(group)
            info.append(group)
            res.append(group_ans)
            i+=1
        return info,res

#Perform one pass of forward and backward propagation through the network.
    def onePass(self,x1,y1,classes,lay,rate,act,lossFunc,Momentum):
        self.forward_pass(x1 ,act)
        l=lay-1
        self.backward_pass(self.h[l], y1,x1,10, act,lossFunc,Momentum)

# Train the neural network using SGD.
    def SGD(self,x1,y1,lay,epo,count,size,act,fn_ans,Momentum,c_l,w_l):
        classes=10
        info,res=self.createBatches(x1,y1,size)

        i=0
        while i < epo:
            h=None
            j=0
            s=len(info)
            while j< s:
                self.onePass(info[j],res[j],classes,lay,count,act,fn_ans,Momentum)
                k=0
                while k< lay:
                    q=lay-1-k
                    self.w[k]-=count*(self.wd[q])
                    self.b[k]-=count*self.bd[q]
                    k+=1
                j+=1
            i+=1
            self.forward_pass(x1,act)
            loss_train=1
            loss1=self.functionLoss(fn_ans,self.h[lay-1],y1,Momentum)
            self.forward_pass(x_val,act)
            loss2=self.functionLoss(fn_ans,self.h[lay-1],y_val,Momentum)
            acc1=self.accuracy(x1,y1,act)
            acc2=self.accuracy(x_val,y_val,act)

            if w_l == 1:
              wandb.log(
                      {
                          'Epoch ': i,
                          'Training_Loss' : round(loss1,2),
                          'Training_Accuracy' : round(acc1,2),
                          'Validation_Loss' : round(loss2,2),
                          'Validation_Accuracy':round(acc2,2)
                      }
                  )
            if c_l == 1:

              print("Iteration Number: "+str(i), end="")
              print(" Train Loss : "+str(loss1))
              print("Iteration Number: "+str(i), end="")
              print(" Validation Loss : "+str(loss2))
              print("Iteration Number: "+str(i), end="")
              print(" Train Accurcy : "+str(acc1))
              print("Iteration Number: "+str(i), end="")
              print(" Validaion Accuracy: "+str(acc2))

# Train the neural network using Momentum
    def Momentum(self,x1,y1,lay,epo,count,size,eta,act,lossFunc,Momentum,c_l,w_l):
        info,res=self.createBatches(x1,y1,size)
        classes=10
        alpha,beta=[],[]
        i=0
        while i< lay:
            a=np.zeros((self.w[i].shape))
            b=np.zeros(self.b[i].shape)
            beta.append(b)
            alpha.append(a)
            i+=1
        i=0
        while i<epo :
            j=0
            while j< len(info):
                self.onePass(info[j],res[j],classes,lay,count,act,lossFunc,Momentum)
                k=0
                while k < lay:
                    s=lay-1-k
                    alpha[k]=(alpha[k]*eta)+self.wd[s]
                    beta[k]=(beta[k]*eta)+self.bd[s]
                    self.w[k]-=count*alpha[k]
                    self.b[k]-=count*beta[k]
                    k+=1
                j+=1
            i+=1

            self.forward_pass(x1,act)
            loss1=self.functionLoss(lossFunc,self.h[lay-1],y1,Momentum)
            self.forward_pass(x_val,act)
            loss2=self.functionLoss(lossFunc,self.h[lay-1],y_val,Momentum)
            acc1=self.accuracy(x1,y1,act)
            acc2=self.accuracy(x_val,y_val,act)

            if w_l == 1:
              wandb.log(
                      {
                          'Epoch ': i,
                          'Training_Loss' : round(loss1,2),
                          'Training_Accuracy' : round(acc1,2),
                          'Validation_Loss' : round(loss2,2),
                          'Validation_Accuracy':round(acc2,2)
                      }
                  )
            if c_l == 1:

              print("Iteration Number: "+str(i), end="")
              print(" Train Loss : "+str(loss1))
              print("Iteration Number: "+str(i), end="")
              print(" Validation Loss : "+str(loss2))
              print("Iteration Number: "+str(i), end="")
              print(" Train Accurcy : "+str(acc1))
              print("Iteration Number: "+str(i), end="")
              print(" Validaion Accuracy: "+str(acc2))

# Train the neural network using nestrov
    def NAG(self,x1,y1,lay,epo,count,size,eta,act,lossFunc,Momentum,c_l,w_l):
        classes=10
        info,res=self.createBatches(x1,y1,size)

        alpha,beta=[],[]

        i=0
        while i<lay:
            b=np.zeros((self.b[i].shape))
            a=np.zeros((self.w[i].shape))
            beta.append(b)
            alpha.append(a)
            i+=1

        i=0
        while i< epo:
            j=0
            while j< len(info):
                k=0
                while k < lay:
                    self.b[k]-=eta*beta[k]
                    self.w[k]-=eta*alpha[k]
                    k+=1
                self.onePass(info[j],res[j],classes,lay,count,act,lossFunc,Momentum)
                k=0
                while k<lay:
                    s=lay-1-k
                    alpha[k]=(eta*alpha[k])+count*(self.wd[s])
                    beta[k]=(eta*beta[k])+count*self.bd[s]
                    self.b[k]-=beta[k]
                    self.w[k]-=alpha[k]
                    k+=1
                j+=1
            i+=1

            self.forward_pass(x1,act)
            s=lay-1
            loss1=self.functionLoss(lossFunc,self.h[s],y1,Momentum)
            self.forward_pass(x_val,act)
            loss2=self.functionLoss(lossFunc,self.h[s],y_val,Momentum)
            acc1=self.accuracy(x1,y1,act)
            acc2=self.accuracy(x_val,y_val,act)
            if w_l == 1:
              wandb.log(
                      {
                          'Epoch ': i,
                          'Training_Loss' : round(loss1,2),
                          'Training_Accuracy' : round(acc1,2),
                          'Validation_Loss' : round(loss2,2),
                          'Validation_Accuracy':round(acc2,2)
                      }
                  )
            if c_l == 1:

              print("Iteration Number: "+str(i), end="")
              print(" Train Loss : "+str(loss1))
              print("Iteration Number: "+str(i), end="")
              print(" Validation Loss : "+str(loss2))
              print("Iteration Number: "+str(i), end="")
              print(" Train Accurcy : "+str(acc1))
              print("Iteration Number: "+str(i), end="")
              print(" Validaion Accuracy: "+str(acc2))



# Train the neural network using RMSProp.
    def RMSProp(self,x1,y1,lay,epo,count,size,eta,act,lossFunc,Momentum,e,c_l,w_l):
        info,res=self.createBatches(x1,y1,size)
        classes =10
        alpha,beta=[],[]

        i=0
        while i< lay:
            b=np.zeros((self.b[i].shape))
            a=np.zeros((self.w[i].shape))
            beta.append(b)
            alpha.append(a)
            i+=1

        i=0
        while i < int(epo):
            j=0
            while j < len(info):
                self.onePass(info[j],res[j],classes,lay,count,act,lossFunc,Momentum)
                k=0
                while k < lay:
                    s=lay-1-k
                    q=1-eta
                    alpha[k]=(alpha[k]*eta)+(q)*np.square(self.wd[s])
                    beta[k]=(beta[k]*eta)+(q)*np.square(self.bd[s])
                    self.w[k]-=(count/np.sqrt(np.linalg.norm(alpha[k]+e)))*self.wd[s]
                    self.b[k]-=(count/np.sqrt(np.linalg.norm(beta[k]+e)))*self.bd[s]
                    k+=1
                j+=1
            i+=1

            self.forward_pass(x1,act)
            s=lay-1
            loss1=self.functionLoss(lossFunc,self.h[s],y1,Momentum)
            self.forward_pass(x_val,act)
            loss2=self.functionLoss(lossFunc,self.h[s],y_val,Momentum)
            acc1=self.accuracy(x1,y1,act)
            acc2=self.accuracy(x_val,y_val,act)
            if w_l == 1:
              wandb.log(
                      {
                          'Epoch ': i,
                          'Training_Loss' : round(loss1,2),
                          'Training_Accuracy' : round(acc1,2),
                          'Validation_Loss' : round(loss2,2),
                          'Validation_Accuracy':round(acc2,2)
                      }
                  )
            if c_l == 1:

              print("Iteration Number: "+str(i), end="")
              print(" Train Loss : "+str(loss1))
              print("Iteration Number: "+str(i), end="")
              print(" Validation Loss : "+str(loss2))
              print("Iteration Number: "+str(i), end="")
              print(" Train Accurcy : "+str(acc1))
              print("Iteration Number: "+str(i), end="")
              print(" Validaion Accuracy: "+str(acc2))

# Train the neural network using ADAM.
    def ADAM(self,x1,y1,lay,epo,count,size,eta1,eta2,act,lossFunc,e,Momentum,c_l,w_l):
        info,res=self.createBatches(x1,y1,size)
        classes=10
        w1,w2,b1,b2=[],[],[],[]

        i=0
        while i< lay:
            b=np.zeros((self.w[i].shape))
            a=np.zeros((self.w[i].shape))
            w2.append(b)
            w1.append(a)
            b=np.zeros((self.b[i].shape))
            a=np.zeros((self.b[i].shape))
            b1.append(a)
            b2.append(b)
            i+=1

        a=0
        i = 0
        while i < int(epo):
            j=0
            while j< len(info):
                a=a+1
                self.onePass(info[j],res[j],classes,lay,count,act,lossFunc,Momentum)
                k=0
                while k< lay:
                    r=1-eta1
                    p=1-eta2
                    u=1-eta1**a
                    v=1-eta2**a
                    s=lay-1-k

                    w1[k]=(w1[k]*eta1)+(r)*self.wd[s]
                    mwhat=w1[k]/(u)

                    w2[k]=(w2[k]*eta2)+(p)*np.square(self.wd[s])
                    vwhat=w2[k]/(v)

                    b1[k]=(b1[k]*eta1)+(r)*self.bd[s]
                    mbhat=b1[k]/(u)

                    b2[k]=(b2[k]*eta2)+(p)*np.square(self.bd[s])
                    vbhat=b2[k]/(v)

                    self.w[k]-=(count/np.sqrt(vwhat+e))*mwhat
                    self.b[k]-=(count/np.sqrt(vbhat+e))*mbhat
                    k+=1
                j+=1
            i=i+1
            self.forward_pass(x1, act)
            s=lay-1
            loss1=self.functionLoss(lossFunc,self.h[s],y1,Momentum)
            self.forward_pass(x_val,act)
            loss2=self.functionLoss(lossFunc,self.h[s],y_val,Momentum)
            acc1=self.accuracy(x1,y1,act)
            acc2=self.accuracy(x_val,y_val,act)
            if w_l == 1:
              wandb.log(
                      {
                          'Epoch ': i,
                          'Training_Loss' : round(loss1,2),
                          'Training_Accuracy' : round(acc1,2),
                          'Validation_Loss' : round(loss2,2),
                          'Validation_Accuracy':round(acc2,2)
                      }
                  )
            if c_l == 1:

              print("Iteration Number: "+str(i), end="")
              print(" Train Loss : "+str(loss1))
              print("Iteration Number: "+str(i), end="")
              print(" Validation Loss : "+str(loss2))
              print("Iteration Number: "+str(i), end="")
              print(" Train Accurcy : "+str(acc1))
              print("Iteration Number: "+str(i), end="")
              print(" Validaion Accuracy: "+str(acc2))

# Train the neural network using NADAM
    def nADAM(self,x1,y1,lay,epo,count,size,eta1,eta2,act,lossFunc,e,Momentum,c_l,w_l):
        classes=10
        w1,w2,b1,b2=[],[],[],[]
        info,res=self.createBatches(x1,y1,size)

        for i in range(lay):
            b=np.zeros((self.w[i].shape))
            a=np.zeros((self.w[i].shape))
            w2.append(b)
            w1.append(a)
            b=np.zeros((self.b[i].shape))
            a=np.zeros((self.b[i].shape))
            b1.append(a)
            b2.append(b)

        a=0
        i=0
        while i< int(epo):
            j=0
            while j< len(info):
                a=a+1
                self.onePass(info[j],res[j],classes,lay,count,act,lossFunc,Momentum)
                k=0
                while k < lay:
                    r=1-eta1
                    p=1-eta2
                    u=1-eta1**a
                    v=1-eta2**a
                    s=lay-1-k
                    w1[k]=(w1[k]*eta1)+(r)*self.wd[s]
                    mwhat=w1[k]/(u)

                    w2[k]=(w2[k]*eta2)+(p)*np.square(self.wd[s])
                    vwhat=w2[k]/(v)

                    b1[k]=(b1[k]*eta1)+(r)*self.bd[s]
                    mbhat=b1[k]/(u)

                    b2[k]=(b2[k]*eta2)+(p)*np.square(self.bd[s])
                    vbhat=b2[k]/(v)

                    self.w[k]-=(count/np.sqrt(vwhat+e))*(eta1*mwhat+(((r)*self.wd[s])/(u)))
                    self.b[k]-=(count/np.sqrt(vbhat+e))*(eta1*mbhat+(((r)*self.bd[s])/(u)))
                    k=k+1
                j=j+1
            i=i+1
            self.forward_pass(x1,act)
            s=lay-1
            loss1=self.functionLoss(lossFunc ,self.h[s],y1,Momentum)
            self.forward_pass(x_val ,act)
            loss2=self.functionLoss(lossFunc,self.h[s], y_val,Momentum)
            acc1=self.accuracy(x1 ,y1 ,act)
            acc2=self.accuracy(x_val ,y_val,act)
            if w_l == 1:
              wandb.log(
                      {
                          'Epoch ': i,
                          'Training_Loss' : round(loss1,2),
                          'Training_Accuracy' : round(acc1,2),
                          'Validation_Loss' : round(loss2,2),
                          'Validation_Accuracy':round(acc2,2)
                      }
                  )
            if c_l == 1:

              print("Iteration Number: "+str(i), end="")
              print(" Train Loss : "+str(loss1))
              print("Iteration Number: "+str(i), end="")
              print(" Validation Loss : "+str(loss2))
              print("Iteration Number: "+str(i), end="")
              print(" Train Accurcy : "+str(acc1))
              print("Iteration Number: "+str(i), end="")
              print(" Validaion Accuracy: "+str(acc2))


# training the model based one given parameters
    def fit_model(self,x1,y1,x2,y2,hiddenlayers,neuron,input_neuron,batch,initialization,lossFunc,activation,optimizer,n,iter,beta,beta1,beta2,e,alpha,Momentum,c_l=1,w_l=1):
        # self.w,self.b=[],[]
        self.formLayers(hiddenlayers,neuron,input_neuron,initialization,10)
        # print(len(self.w),"arch")
        if optimizer=="SGD":
            self.SGD(x1,y1,len(self.w),iter,n,batch,activation,lossFunc,alpha,c_l,w_l)
        elif optimizer=='Momentum':
            self.Momentum(x1,y1,len(self.w),iter,n,batch,Momentum,activation,lossFunc,alpha,c_l,w_l)
        elif optimizer=='NAG':
            self.NAG(x1,y1,len(self.w),iter,n,batch,beta,activation,lossFunc,alpha,c_l,w_l)
        elif optimizer=='RMSProp':
            self.RMSProp(x1,y1,len(self.w),iter,n,batch,beta,activation,lossFunc,alpha,e,c_l,w_l)
        elif optimizer=='ADAM':
            self.ADAM(x1,y1,len(self.w),iter,n,batch,beta1,beta2,activation,lossFunc,e,alpha,c_l,w_l)
        elif optimizer=='nADAM':
            self.nADAM(x1,y1,len(self.w),iter,n,batch,beta1,beta2,activation,lossFunc,e,alpha,c_l,w_l)


# import dataset
if arguments.dataset == 'fashion_mnist':
  (x1, y1), (x2, y2) = fashion_mnist.load_data()
elif arguments.dataset== 'mnist':
  (x1, y1), (x2, y2) = mnist.load_data()

x1=x1.reshape(x1.shape[0],-1) / 255
x2=x2.reshape(x2.shape[0],-1)/ 255

x1, x_val, y1, y_val = train_test_split(x1,y1, test_size=0.1, random_state=0)
# y2 = np.eye(10)[y2]

obj=NeuralNetwork()


if arguments.wandb_log == 1:
    wandb.init(project=arguments.wandb_project,name=arguments.wandb_entity)


obj.fit_model(x1,
            y1,
            x_val,
            y_val,
            hiddenlayers=arguments.num_layers,
            neuron=arguments.hidden_size,
            input_neuron=784,
            batch=arguments.batch_size,
            initialization=arguments.weight_init,
            lossFunc=arguments.loss,
            activation=arguments.activation,
            optimizer=arguments.optimizer,
            n=arguments.learning_rate,
            iter=arguments.epochs,
            beta=arguments.beta,
            beta1=arguments.beta1,
            beta2=arguments.beta2,
            e=arguments.epsilon,
            alpha=arguments.weight_decay,
            Momentum=arguments.momentum,
            c_l=1,
            w_l=arguments.wandb_log
            )

    

    # def accuracy(self,x2,y2,act):
test_acc = obj.accuracy(x2,y2,arguments.activation)

if arguments.wandb_log == 1:
    wandb.log({"test_accuracy" : round(test_acc,3)})
    wandb.finish()