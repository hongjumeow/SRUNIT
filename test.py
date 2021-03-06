"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from PIL import Image
import numpy as np
from tqdm import tqdm # library to show progress
from scipy.linalg import sqrtm
from torchvision import transforms

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options

    base_path = os.path.join(opt.output_path, '%s:%s_infer_results' % (opt.name, opt.epoch))
    os.makedirs(base_path, exist_ok=True)

    for i, data in tqdm(enumerate(dataset)): 
        if i == 0:
            model.data_dependent_initialize(data, infer_mode=True)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                # model.eval()
                state = engine.run()
        if i == opt.num_test: break  # only apply our model to opt.num_test images.
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        p = img_path[0].split('/')[-1]

        print(model.compute_fid())
        
        # save imgs

        generated_img = visuals['fake_B'].cpu().numpy()[0]
        generated_img = ((np.transpose(generated_img, [1, 2, 0]) + 1) * 127.5).astype(np.uint8)
        
        real_img = visuals['real_A'].cpu().numpy()[0]
        real_img = ((np.transpose(real_img, [1, 2, 0]) + 1) * 127.5).astype(np.uint8)

        if(opt.save_img_mode != 'alone'):
            list_imgs = [ real_img, generated_img ]
            imgs_comb = []
            if(opt.save_img_mode == 'vertical'):
                imgs_comb = np.vstack(list_imgs)
            else :
                imgs_comb = np.hstack(list_imgs)
            imgs_comb = Image.fromarray(imgs_comb)
            imgs_comb.save(os.path.join(base_path, p))
        else :
            Image.fromarray(generated_img).save(os.path.join(base_path, p))
