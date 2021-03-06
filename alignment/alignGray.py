from absl import app, flags, logging
from absl.flags import FLAGS
import os
import numpy as np
import imageio
import tqdm
import rawpy

flags.DEFINE_string('dirpath', '', 'file path')

def main(_):
    dirpath = FLAGS.dirpath
    my_files = [f.name for f in os.scandir(dirpath) if
                f.name.endswith('.raw')]
    for one in tqdm.tqdm(my_files):
        raw2 = np.fromfile(os.path.join(dirpath, one),
                           dtype=np.uint8)
        raw2 = np.reshape(raw2, (1024, 1280, 1))
        outfile = one[:-4]
        imageio.imsave(
            os.path.join(dirpath, outfile + '.tif'), raw2)
        cmd_str = "exiftool   -@ pbpx_exft_args.txt   -o %(outputdngname)s  %(tifname)s" % {
            "outputdngname": dirpath + outfile + '.dng',
            "tifname": dirpath + outfile + '.tif'}
        a = os.system(cmd_str)
        with rawpy.imread(
                dirpath + outfile + '.dng') as raw:
            rgbimg = raw.postprocess(use_camera_wb=True,
                                     half_size=False,
                                     no_auto_bright=True,
                                     output_bps=8)# 8 bits
        # img = Image.fromarray(rgbimg)
        # display(img)
        imageio.imsave(dirpath + outfile + '.jpg', rgbimg)

if __name__ == '__main__':
    app.run(main)
