from pulse2percept.models import AxonMapModel
from pulse2percept.implants import ElectrodeGrid, ProsthesisSystem
from pulse2percept.stimuli import ImageStimulus
from matplotlib import pyplot as plt
import cv2
import numpy as np

SPACE = 300

def build_p2p_model_and_implant(size, axlambda=100, rho=150, range_limit=13):
    p2p_model = AxonMapModel(
        axlambda=axlambda, rho=rho, 
        xrange=(-range_limit, range_limit), yrange=(-range_limit, range_limit), xystep=1
    )
    p2p_model.build()

    square16_implant = ProsthesisSystem(earray=ElectrodeGrid((size, size), SPACE))
    
    return p2p_model, square16_implant


p2p_model, square16_implant = build_p2p_model_and_implant(size=14, axlambda=1000, rho=500, range_limit=13)

# img = cv2.imread('/Users/yuli/Downloads/n03026506_2749_stim.jpg', 0)
img = cv2.imread('/home/students/tnguyen/masterthesis/plots/437_1420/train/n03026506/n03026506_2749_stim.jpg', 0)
print(img.shape, np.max(img), np.min(img))


# just check the "visual field" with four corners. 
# cornor = np.zeros((14,14))
# cornor[0,0] = 1.
# cornor[0,13] = 1.
# cornor[13,0] = 1.
# cornor[13,13] = 1.
# square16_implant.stim = cornor.flatten()
# percept = p2p_model.predict_percept(square16_implant)
# ax = percept.plot()
# plt.show()
        
percept_full = np.zeros(img.shape)
for x in range(16):
    for y in range(16):
        patch = img[x*14:x*14+14, y*14:y*14+14]

        # Optional: normalize the patch to [-1, 1]
        # patch =2.0* patch / 255.0 - 1.0

        # normalize the patch to [0, 1]
        # patch = patch / 255.0
        print(np.max(patch), np.min(patch))

        # take notes of the min and max of the patch
        patch_min = np.min(patch)
        patch_max = np.max(patch)

        square16_implant.stim = patch.flatten()
        percept = p2p_model.predict_percept(square16_implant)
        
        percept_crop = percept.data[6:20, 6:20].squeeze()
        # the value range of the percept is always weird 
        # it's better to normalize to the range as the original patch
        print(np.max(percept_crop), np.min(percept_crop))

        # from sklearn.preprocessing import minmax_scale
        # percept_crop_reranged = minmax_scale(percept_crop.flatten(), feature_range=(patch_min, patch_max)).reshape(14,14)

        # now the range is correct
        # print(np.max(percept_crop_reranged), np.min(percept_crop_reranged))

        # percept_full[x*14:x*14+14, y*14:y*14+14] = percept_crop_reranged
        percept_full[x*14:x*14+14, y*14:y*14+14] = percept_crop

plt.imshow(percept_full, cmap='gray')
plt.show()
