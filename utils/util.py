import torch
import paddle.fluid as fluid

resnet18 = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'


def torch_weight_to_paddle_model(torch_weight_file, paddle_model):
    torch_weight = torch.load(torch_weight_file)
    fluid.dygraph.save_dygraph(paddle_model.state_dict(), './pretrained/resnet18-torch')
    paddle_weight, _ = fluid.dygraph.load_dygraph('./pretrained/resnet18-torch')
    for k, p in torch_weight.items():
        if k in paddle_weight:
            np_parm = torch_weight[k].detach().numpy()
            if np_parm.shape == paddle_weight[k].shape:
                paddle_weight[k] = np_parm
            else:
                print('torch parm {} dose not match paddle parm {}'.format(k, k))
        elif 'running_mean' in k:
            np_parm = torch_weight[k].detach().numpy()
            if np_parm.shape == paddle_weight[k[:-12]+'_mean'].shape:
                paddle_weight[k[:-12]+'_mean'] = np_parm
            else:
                print('torch parm {} dose not match paddle parm {}'.format(k, k[:-12]+'_mean'))
        elif 'running_var' in k:
            np_parm = torch_weight[k].detach().numpy()
            if np_parm.shape == paddle_weight[k[:-11] + '_variance'].shape:
                paddle_weight[k[:-11] + '_variance'] = np_parm
            else:
                print('torch parm {} dose not match paddle parm {}'.format(k, k[:-11] + '_variance'))
        else:
            print('torch parm {} not exist in paddle modle'.format(k))
    paddle_model.set_dict(paddle_weight)
    fluid.dygraph.save_dygraph(paddle_model.state_dict(), './pretrained/resnet18-torch')


def model_size(model):
    total_num = sum(p.numpy().size for p in model.parameters())
    trainable_num = sum(p.numpy().size for p in model.parameters() if not p.stop_gradient)
    print('Total: {:.5f}M  Trainable: {:.5f}M'.format(total_num / 1e6, trainable_num / 1e6))