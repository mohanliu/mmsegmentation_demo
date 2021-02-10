from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

config_file = 'config.py'
checkpoint_file = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

config = mmcv.Config.fromfile(config_file)
print(config.keys())

model = init_segmentor(config, checkpoint_file, device='cuda:1')


# test a single image
img = 'sample/demo.png'
result = inference_segmentor(model, img)

