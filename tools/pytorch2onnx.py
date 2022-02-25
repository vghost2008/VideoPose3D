import os.path as osp
import numpy as np
import torch
from common.model import TemporalModel
from onnxsim import simplify


try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')

try:
    from mmcv.onnx.symbolic import register_extra_symbolics
except ModuleNotFoundError:
    raise NotImplementedError('please update mmcv to version>=1.0.4')


def _convert_batchnorm(module):
    """Convert the syncBNs into normal BN3ds."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm3d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def pytorch2onnx(model,
                 input_shape,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False):
    """Convert pytorch model to onnx model.

    Args:
        model (:obj:`nn.Module`): The pytorch model to be exported.
        input_shape (tuple[int]): The input tensor shape of the model.
        opset_version (int): Opset version of onnx used. Default: 11.
        show (bool): Determines whether to print the onnx model architecture.
            Default: False.
        output_file (str): Output onnx model name. Default: 'tmp.onnx'.
        verify (bool): Determines whether to verify the onnx model.
            Default: False.
    """
    model.cpu().eval()

    input_tensor = torch.randn(input_shape)

    '''torch.onnx.export(
        model,
        input_tensor,
        output_file,
        export_params=True,
        keep_initializers_as_inputs=True,
        verbose=show)'''
    torch.onnx._export(
        model,
        input_tensor,
        output_file,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {1: 'N'},
                      'output': {1: 'N'}},
        opset_version=11,
        )
    '''dynamic_axes = {'input_1':[1],
                         'output':[1]})
                         'output':[1]})'''
    
    onnx_model = onnx.load(output_file)
    model, check = simplify(onnx_model,
                            dynamic_input_shape=True,
                            input_shapes={'input':input_shape})
    onnx.save(model, output_file)
    print(f'Successfully exported ONNX model: {output_file}')
    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # check the numerical value
        # get pytorch output
        pytorch_result = model(input_tensor)[0].detach().numpy()

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert len(net_feed_input) == 1
        sess = rt.InferenceSession(output_file)
        onnx_result = sess.run(
            None, {net_feed_input[0]: input_tensor.detach().numpy()})[0]
        # only compare part of results
        random_class = np.random.randint(pytorch_result.shape[1])
        assert np.allclose(
            pytorch_result[:, random_class], onnx_result[:, random_class]
        ), 'The outputs are different between Pytorch and ONNX'
        print('The numerical values are same between Pytorch and ONNX')


if __name__ == '__main__':

    ckpt_pos = 'weights_semv5/epoch_100.bin'
    in_features = 3
    filter_widths = [3,3,3,3]
    model_pos = TemporalModel(num_joints_in=17,
                    in_features=in_features,num_joints_out=17,
                    filter_widths=filter_widths,
                    causal=False,
                    channels=1024,dense=False)
    checkpoint = torch.load(ckpt_pos)

    model_pos.load_state_dict(checkpoint['model_pos'])
    model_pos.eval()

    pytorch2onnx(
        model_pos,
        [1,300,17,3],
        show=True,
        output_file="model.onnx",
        verify=False)
