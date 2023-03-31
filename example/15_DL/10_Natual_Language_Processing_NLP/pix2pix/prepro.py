import cv2
import glob
dir = 'seg_train'
paths = glob.glob('./data/{}/*/*.jpg'.format(dir))
cnt = 0
for path in paths:
    cnt+=1

    path = str(path)
    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    spath = './data/{}/' + path.split('\\')[-1]
    cv2.imwrite(spath.format('ori_train'), img)
    cv2.imwrite(spath.format('gray_train'), gray)


dir = 'seg_test'
paths = glob.glob('./data/{}/*/*.jpg'.format(dir))
cnt = 0
for path in paths:
    cnt+=1

    path = str(path)
    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    spath = './data/{}/' + path.split('\\')[-1]
    cv2.imwrite(spath.format('ori_test'), img)
    cv2.imwrite(spath.format('gray_test'), gray)
