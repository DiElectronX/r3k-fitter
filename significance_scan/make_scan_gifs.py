import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from PIL import Image
import fitz
import io

def pdf_to_animation(pdf_files, output_file='animation.mp4', fps=4):
    images = []

    # Convert PDF pages to images
    for pdf_file in pdf_files:
        doc = fitz.open(pdf_file)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            images.append(img)

    # Create animation
    fig, ax = plt.subplots()
    ims = []

    for img in images:
        im = ax.imshow(img, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=1000/fps, blit=True, repeat_delay=1000)

    # Save the animation
    ani.save(output_file, writer='pillow')


def make_scan_gif(indir, path):
    jpsi_files = np.array([f for f in indir.glob('*jpsi_fit*')])
    jpsi_files_sorted = jpsi_files[np.argsort([float(str(f).split('>')[1].rsplit('.', 1)[0]) for f in jpsi_files])]
    pdf_to_animation(jpsi_files_sorted, output_file='jpsi_animation.gif')
    
    lowq2_files = np.array([f for f in indir.glob('*lowq2_fit*')])
    lowq2_files_sorted = lowq2_files[np.argsort([float(str(f).split('>')[1].rsplit('.', 1)[0]) for f in lowq2_files])]
    pdf_to_animation(lowq2_files_sorted, output_file='lowq2_animation.gif')
    
def main(args):
    if args.input_dir:
        input_dir = Path(args.input_dir)
        assert input_dir.is_dir(), 'Cannot find input directory'
    else:
        input_dir = Path('.') / 'scan_fits'
        assert input_dir.is_dir(), 'Cannot find input directory'

    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path('.') / 'significance_scan.gif'

    if args.label:
        output_file = output_file.with_stem('_'.join([str(output_file.stem), args.label]))

    make_scan_gif(input_dir, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', dest='input_dir', 
        type=str, help='input directory')
    parser.add_argument('-o', '--output', dest='output', 
        type=str, help='output file path')
    parser.add_argument('-l', '--label', dest='label', 
        type=str, help='output file label')
    args, _ = parser.parse_known_args()

    main(args)
