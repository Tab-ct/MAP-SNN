import torch
import torch.nn as nn
import torch.nn.functional as F

# Synaptic Model using 1-D convolution
class kernel_conv1d_MAP_SNN(nn.Module):
    def __init__(self,chanels, kernel_size=7, device='cpu'):
        super(kernel_conv1d_MAP_SNN, self).__init__()

        # learnable parameters
        self.kernel_parameter_a = nn.Parameter((torch.rand(chanels, device=device)* 0.5 + 0.5).unsqueeze(1) ) # shape-param a
        self.kernel_parameter_b = nn.Parameter((torch.rand(chanels, device=device)* 0.5 + 0.5).unsqueeze(1) ) # shape-param b
        self.kernel_time_shift = nn.Parameter(torch.ones(chanels, device=device).float() * 0.8)     # delay-param delay
        
        # time interval [1,kernel_size]
        kernel_time = torch.arange(1,kernel_size+1).to(device)
        kernel_time = torch.flip(kernel_time, dims=[0])
        self.kernel_time = kernel_time.expand(chanels,kernel_size)

        # convolution setting
        self.kernel_size = kernel_size
        self.chanels = chanels
        self.padding = int((kernel_size-1)/2)

        # derive time matric for calculation
        self.kernel_time_shift_align = self.kernel_time_shift.expand(kernel_size,chanels).T


    def forward(self,input_tensor):
        ############### Tensor-Forwarding ###############
        ## size = (batch_size, time_window, n_neuron) ###
        #################################################
        input_tensor = input_tensor.permute(0,2,1)
        kernel_time_shifted = F.relu(self.kernel_time - F.relu(self.kernel_time_shift_align))
        kernel_weight = torch.exp(-(kernel_time_shifted * F.relu(self.kernel_parameter_a))) - torch.exp(-(kernel_time_shifted * F.relu(self.kernel_parameter_b)))
        kernel_weight = kernel_weight.reshape(self.chanels,1,self.kernel_size)
        out = F.conv1d(input_tensor,kernel_weight,padding=self.padding, groups=self.chanels)
        return out.permute(0,2,1)



# Activation function used in LIF neurons
class ActFun_MAP_SNN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mem, n_rf):
        n_spikes = mem.gt(0).float() * (n_rf).floor().float()
        ctx.save_for_backward(mem)
        return n_spikes

    @staticmethod
    def backward(ctx, grad_output):
        mem, = ctx.saved_tensors
        grad_mem = None
        grad_n_rf =  mem.gt(0).float() * grad_output
        return grad_mem, grad_n_rf

act_fun_map_snn = ActFun_MAP_SNN.apply

# Spiking Neural Model under Multiple-Spike Pattern
class LIF_neurons_comb_time_MAP_SNN(nn.Module):
    def __init__(self, n_neurons, device=torch.device("cpu")):
        super(LIF_neurons_comb_time_MAP_SNN, self).__init__()
        
        # model init
        self.n_neurons = n_neurons
        self.device = device
        self.h_mem = None
        self.h_spike = None
        self.act_fun = act_fun_map_snn

        # neuron parameters
        self.h_thresh = torch.ones(self.n_neurons, device=self.device).float() * 2  # threshold         $V_{threshold}$
        self.h_inh = torch.ones(self.n_neurons, device=self.device).float() * 1.2   # inhibition factor $q$
        self.h_decay = nn.Parameter(torch.ones(self.n_neurons, device=self.device).float() * 0.2)   # learnable decay factor

    # Forward once among time
    def forward_single_time(self,input):
        self.h_mem, self.h_spike = self.mem_update(input, self.h_mem, self.h_spike, self.act_fun)
        return self.h_spike

    # Forward all times
    def forward(self,input):
        bsz,T= input.shape[0],input.shape[1]
        self.init_mem(bsz)
        out_spike = torch.zeros(bsz,T,input.shape[-1]).to(self.device)
        for t in range(T):
            self.h_mem, self.h_spike = self.mem_update(input[:,t,:], self.h_mem, self.h_spike)
            out_spike[:,t,:] = self.h_spike
        return out_spike
    
    # State Init
    def init_mem(self, batch_size):
        self.h_mem = torch.zeros(batch_size, self.n_neurons, device=self.device)
        self.h_spike = torch.zeros(batch_size, self.n_neurons, device=self.device)
    
    # State Update
    def mem_update(self, input, mem, spike):
        mem_temp = mem + input
        thresh = self.h_thresh
        decay = (F.relu6(self.h_decay * 6) / 6) # strict in [0,1]
        n_rf = (torch.log(F.relu(mem_temp) / thresh * (self.h_inh - 1) + 1) / torch.log(self.h_inh)).float()
        spike = self.act_fun(mem_temp, n_rf) 
        mem_reset = (self.h_inh ** spike - 1) / (self.h_inh - 1) * thresh
        mem_ret =(mem_temp - mem_reset) * decay
        return mem_ret, spike






################################################################
########           Models for Dataset: SHD              ########
################################################################
class SNN_model_for_SHD_MAP_SNN(nn.Module):
    def __init__(self, device='cpu', n_input=700, n_class=20):
        super(SNN_model_for_SHD_MAP_SNN, self).__init__()
        
        # model init
        self.device = device
    
        # define layers of LIF neurons
        self.h1 = LIF_neurons_comb_time_MAP_SNN(400, device=device) 
        self.h2 = LIF_neurons_comb_time_MAP_SNN(400, device=device) 
        self.h3 = LIF_neurons_comb_time_MAP_SNN(20,  device=device) 

        # convolutional synapsis after LIF-neurons
        self.conv1 = kernel_conv1d_MAP_SNN(400, device=device)
        self.conv2 = kernel_conv1d_MAP_SNN(400, device=device)

        # network setup
        self.net1 = nn.Sequential(
            nn.Linear(n_input, 400, bias = True),
            self.h1,
            # self.conv1,
            nn.Linear(400, 400, bias = True),
            self.h2,
            # self.conv2,
            nn.Linear(400, n_class, bias = True),
            self.h3,
        )
    def forward(self, input):

        ############### Tensor-Forwarding ###############
        ## size = (batch_size, time_window, n_neuron) ###
        #################################################

        # network forwarding
        out1 = self.net1(input)

        # take sum of spikes as prediction
        outputs = out1.sum(1)

        return outputs




################################################################
##########        Models for Dataset: N-MNIST         ##########
################################################################

class SNN_model_for_N_MNIST_MAP_SNN(nn.Module):
    def __init__(self, device='cpu', n_input=32*32, n_class=10):
        super(SNN_model_for_N_MNIST_MAP_SNN, self).__init__()
    
        # model init
        self.device = device
        self.n_hidden = 800
        self.each_hidden_num = int(self.n_hidden/2)
    
        # define layers of LIF neurons
        self.h1_1 = LIF_neurons_comb_time_MAP_SNN(self.each_hidden_num, device=device) 
        self.h1_2 = LIF_neurons_comb_time_MAP_SNN(self.each_hidden_num, device=device) 
        self.h2 = LIF_neurons_comb_time_MAP_SNN(n_class, device=device) 

        # convolutional synapsis after LIF-neurons
        self.conv1 = kernel_conv1d_MAP_SNN(self.each_hidden_num*2, device=device)

        # network setup
        self.net1_1 = nn.Sequential(
            nn.Linear(n_input, self.each_hidden_num, bias = True),
            self.h1_1,
        )
        self.net1_2 = nn.Sequential(
            nn.Linear(n_input, self.each_hidden_num, bias = True),
            self.h1_2,
        )
        self.net2 = nn.Sequential(
            # self.conv1,
            nn.Linear(self.each_hidden_num*2, n_class, bias = True),
            self.h2,
        )
        
    def forward(self, input):

        ############### Tensor-Forwarding ###############
        ## size = (batch_size, time_window, n_neuron) ###
        #################################################

        ##### Input Size: [batch_size, time_steps, channel, size, size] #######
        batch_size = input.shape[0]
        time_window = input.shape[1]
        input1, input2 = torch.chunk(input, 2, dim=2)
        input1 = input1.reshape(batch_size, time_window, -1)
        input2 = input2.reshape(batch_size, time_window, -1)

        # network forwarding
        h_out1 = self.net1_1(input1)
        h_out2 = self.net1_2(input2)
        h_out = torch.cat([h_out1, h_out2], dim = 2)
        out = self.net2(h_out)

        # take sum of spikes as prediction
        outputs = out.sum(1)

        return outputs