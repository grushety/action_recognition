import scipy.io

PATH = "/home/yulia/pepper_ws/src/action_recognition/scripts/learning"

data = scipy.io.loadmat(PATH + "/database/test_data.mat")
test = 1 * data["data"]
print test.shape
for i in test:
    print i

