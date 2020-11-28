from PIL import Image
import os


def png2jpg(image_path):
    infile = image_path
    outfile = os.path.splitext(infile)[0] + ".jpg"
    img = Image.open(infile)
    try:
        if len(img.split()) == 4:
            # prevent IOError: cannot write mode RGBA as BMP
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            img.convert('RGB').save(outfile, quality=95)
            # os.remove(image_path)
        else:
            img.convert('RGB').save(outfile, quality=95)
            # os.remove(image_path)
        return outfile
    except Exception as e:
        print("PNG转换JPG 错误", e)


def batch_convert(root_dir):
    img_dir = os.listdir(root_dir)
    for img_sample in img_dir:
        if img_sample.endswith('.png'):
            temp_file = root_dir + img_sample
            png2jpg(temp_file)
    print("Finish!")


if __name__ == "__main__":
    target_file = r"C:\Users\admin\Documents\data\test\v3_3.png"
    png2jpg(target_file)
