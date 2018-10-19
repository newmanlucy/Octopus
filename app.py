import cv2
import os

raw_folder = 'raw_im'
bw_folder = 'bw_im'

def convert_to_bw(filename):
	path = os.path.join(raw_folder, filename)

	img = cv2.imread(path)

	# Converts to black and white with 3 channels (I THINK)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# print(img_gray.shape) # (178,178,3) for 3 channels?

	out_path = os.path.join(bw_folder, filename)
	cv2.imwrite(out_path, img_gray)



if __name__ == '__main__':
	convert_to_bw("6GXY9rCA.jpeg")