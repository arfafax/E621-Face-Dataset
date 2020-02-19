import pandas as pd
import wget
import os.path
import cv2
from os import path
import argparse

settings = {
        'IMAGE_PATH'     :  'out',
        'CROP_PATH'      :  'crop',
        'CROP_SIZE'      :  512,
        'BORDER_MODE'    :  cv2.BORDER_CONSTANT,
        'SQUARE'         :  False,
        'MIN_SCORE'      :  10,
        'MIN_CONFIDENCE' :  0.99
}

def get_file(row):
    print("\n\n")
    print("Downloading", row['e621id'], row['file_url'])
    try:
        url = row['file_url']
        filename = row['file_url'].split('/')[-1]
        if path.exists(settings['IMAGE_PATH'] + "/" + filename):
            print("Exists", filename)
            return 0
        wget.download(row['file_url'], settings['IMAGE_PATH'] + "/" + filename)
        return 1
    except Exception as e:
        print("Error getting file", row['file_url'], e)
        return 0

def pad_img(img):
    size = settings['CROP_SIZE']
    border = settings['BORDER_MODE']
    old_size = img.shape[:2]
    ratio = float(size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = size - new_size[1]
    delta_h = size - new_size[0]

    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, border, value=[0,0,0])

    return new_im


def crop_img(filename, xmin, ymin, xmax, ymax):
    require_square = settings['SQUARE']
    img = cv2.imread(filename)

    if require_square:
        w = xmax - xmin
        h = ymax - ymin
        new_dim = max(w, h)
        center_x, center_y = (xmax - w//2), (ymax - h//2)
        xmin = max(0, center_x - new_dim//2)
        xmax = min(img.shape[1], center_x + new_dim//2)
        ymin = max(0, center_y - new_dim//2)
        ymax = min(img.shape[0], center_y + new_dim//2)

    cropped_img = img[ymin:ymax, xmin:xmax]
    return pad_img(cropped_img)

def crop(row):
    filename = row['file_url'].split('/')[-1]
    out_name = ".".join([str(row['e621id']), str(row['index']), 'png'])

    if path.exists(settings['IMAGE_PATH'] + "/" + filename):
        print("Cropping id %s, face %s" % (row['e621id'], row['index']))
        cv2.imwrite(settings['CROP_PATH'] + "/" + out_name, crop_img(settings['IMAGE_PATH'] + "/" + filename,
                                                   int(row['xmin']),
                                                   int(row['ymin']),
                                                   int(row['xmax']),
                                                   int(row['ymax'])))

#==========================================================
def download_images(faces):
    if not os.path.exists(settings['IMAGE_PATH']):
        os.mkdir(settings['IMAGE_PATH'])

    print("Getting %d images (%d bytes)" % (faces.shape[0], faces['file_size'].sum()))
    faces.apply(get_file, axis=1)
#==========================================================
def crop_images(faces):
    if not os.path.exists(settings['CROP_PATH']):
        os.mkdir(settings['CROP_PATH'])
    faces.apply(crop, axis=1)
#==========================================================
_examples = '''examples:
    # Download faces with score >= 10
    python %(prog)s download --output-dir=out --min-score=10
'''
#==========================================================
def main():
    parser = argparse.ArgumentParser(
            description='''e621 Portrait downloader.''',
            epilog=_examples,
            formatter_class=argparse.RawDescriptionHelpFormatter
            )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    download_images_parser = subparsers.add_parser('download', help='Download uncropped images.')
    download_images_parser.add_argument('--download-dir', help='Root directory for downloaded images (default: %(default)s)', default='out', metavar='DIR')
    download_images_parser.add_argument('--csv', help='CSV file containing faces to crop (default: %(default)s)', default='faces_s.csv', metavar="csv-file", type=argparse.FileType("rt"), required=True)
    download_images_parser.add_argument('--min-score', help='Minimum e621 rating (default: %(default)s)', default=10)
    download_images_parser.add_argument('--min-confidence', help='Minimum detection confidence (default: %(default)s)', default=0.99)
    download_images_parser.add_argument('--species', help='Filter by species tag')
    download_images_parser.add_argument('--copyright', help='Filter by copyright tag')

    crop_images_parser = subparsers.add_parser('crop', help='Crop downloaded images.')
    crop_images_parser.add_argument('--download-dir', help='Root directory for downloaded images (default: %(default)s)', default='out', metavar='DIR')
    crop_images_parser.add_argument('--crop-dir', help='Root directory for cropped images (default: %(default)s)', default='crop', metavar='DIR')
    crop_images_parser.add_argument('--csv', help='CSV file containing faces to crop (default: %(default)s)', default='faces_s.csv', metavar="csv-file", type=argparse.FileType("rt"), required=True)
    crop_images_parser.add_argument('--min-score', help='Minimum e621 rating (default: %(default)s)', default=10)
    crop_images_parser.add_argument('--min-confidence', help='Minimum detection confidence (default: %(default)s)', default=0.99)
    crop_images_parser.add_argument('--crop-size', help='Width/height of cropped image (default: %(default)s)', default=512)
    crop_images_parser.add_argument('--replicate-border', help='Replicate border instead of black border', action='store_true')
    crop_images_parser.add_argument('--square', help='Force images to be square by cropping to maximum of width/height', action='store_true')
    crop_images_parser.add_argument('--species', help='Filter by species tag')
    crop_images_parser.add_argument('--copyright', help='Filter by copyright tag')

    args = parser.parse_args()
    subcmd = args.command
    if subcmd is None:
        print('Error: missing subcommand. Re-run with --help for usage.')
        sys.exit(1)

    kwargs = vars(args)

    csv_path = kwargs.pop('csv')

    features = pd.read_csv(csv_path, encoding='utf-8')
    faces = features[features['feature'] == 'face']

    min_score = int(kwargs.pop('min_score'))
    if min_score:
        faces = faces[faces['score'] >= min_score]

    min_confidence = float(kwargs.pop('min_confidence'))
    if min_confidence:
        faces = faces[faces['confidence'] >= min_confidence]

    species = kwargs.pop('species')
    if species:
        faces = faces[faces.species.str.contains(species, na=False)]

    copyright = kwargs.pop('copyright')
    if copyright:
        faces = faces[faces.copyrights.str.contains(copyright, na=False)]

    if subcmd == 'download':
        settings['IMAGE_PATH'] = kwargs.pop('download_dir')
        download_images(faces)

    if subcmd == 'crop':
        settings['IMAGE_PATH'] = kwargs.pop('download_dir')
        settings['CROP_PATH'] = kwargs.pop('crop_dir')
        settings['CROP_SIZE'] = int(kwargs.pop('crop_size'))

        replicate = kwargs.pop('replicate_border')
        settings['BORDER_MODE'] = cv2.BORDER_CONSTANT
        if replicate:
            settings['BORDER_MODE'] = cv2.BORDER_REPLICATE
        require_square = kwargs.pop('square')
        if require_square:
            settings['SQUARE'] = True

        crop_images(faces)

#==========================================================
if __name__ == "__main__":
    main()
