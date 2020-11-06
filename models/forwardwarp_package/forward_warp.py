import torch
from torch.nn import Module, Parameter
from torch.autograd import Function

import forward_warp_cuda

class forward_warp_function(Function):

    @staticmethod
    def forward(ctx, im0, flow):
        '''
        im0: the first image with shape [B, C, H, W]
        flow: the optical flow with shape [B, H, W, 2] (different to grid_sample, it's range is from [-W, -H] to [W, H])
        '''
        assert(len(im0.shape) == len(flow.shape) == 4)
        assert(im0.shape[0] == flow.shape[0])
        assert(im0.shape[-2:] == flow.shape[1:3])
        assert(flow.shape[3] == 2)
        
        im0 = im0.contiguous()
        flow = flow.contiguous()
        ctx.save_for_backward(im0, flow)

        im1 = torch.zeros(im0.size(), dtype=im0.dtype, layout=im0.layout, device=im0.device)

        # with torch.cuda.device_of(im0):
        forward_warp_cuda.forward(im0, flow, im1)

        return im1

    @staticmethod
    def backward(ctx, grad_output):

        grad_output = grad_output.contiguous()
        im0, flow = ctx.saved_variables
        im0_grad = torch.zeros(im0.size(), dtype=im0.dtype, layout=im0.layout, device=im0.device)
        flow_grad = torch.zeros(flow.size(), dtype=flow.dtype, layout=flow.layout, device=flow.device)

        #with torch.cuda.device_of(im0):
        forward_warp_cuda.backward(grad_output, im0, flow, im0_grad, flow_grad)

        return im0_grad, flow_grad


class forward_warp(Module):

    def __init__(self):
        super(forward_warp, self).__init__()
        
    def forward(self, im0, flow):

        _, _, h, w = im0.size()
        flow = torch.clamp(flow, -2*w, 2*w)

        return forward_warp_function.apply(im0, flow)

### forward warping considering occlusion (further point has less weight)
class ForwardWarpDWeight(Module):
    def __init__(self, ref_scale=5):
        super(ForwardWarpDWeight, self).__init__()

        self.ref_scale = ref_scale  # the weight changes 2.71(e)x when depth is changed ref_scale meters. 

    def forward(self, x, flow, depth):

        _, _, h, w = x.size()
        flow = torch.clamp(flow, -2*w, 2*w)
        flow = flow.transpose(1, 2).transpose(2, 3)

        ### calculate depth weight and forward the weight
        depth = depth.clamp(min=1e-3, max=80)
        depth_weight = torch.exp(-(depth-40)/self.ref_scale) # 8 is just a scaling for numerical stability, does not change result. 80m -> e^(-16)=1.125e-7, e^(-16+8)=3.35e-4
        depth_weight_flowed = forward_warp_function.apply(depth_weight, flow)

        ### forward weighted x
        x_weighted = x * depth_weight
        x_weighted_flowed = forward_warp_function.apply(x_weighted, flow)

        ### zero out empty x
        mask = torch.ones_like(depth)
        mask_flowed = forward_warp_function.apply(mask, flow)

        mask_flowed_invalid = mask_flowed < 0.5
        zero_x = torch.zeros_like(x)
        x_weighted_flowed = torch.where(mask_flowed_invalid, zero_x, x_weighted_flowed)

        assert (depth_weight_flowed[~mask_flowed_invalid] > 1e-7).all()
        ### normalize forward x using forward weighted x and forward x
        x_flowed_normalized = x_weighted_flowed / torch.clamp(depth_weight_flowed, min=1e-7)

        return x_flowed_normalized