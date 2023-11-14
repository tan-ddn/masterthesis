import glob
import os
import matplotlib.pyplot as plt
from pathlib import Path
from numpy import pi
from pulse2percept.models import AxonMapModel
from pulse2percept.implants import ElectrodeGrid, ProsthesisSystem
from pulse2percept.stimuli import ImageStimulus

image_dir = r'/work/scratch/tnguyen/images/cocosearch/patches/'
images = glob.glob(os.path.join(image_dir, "*/*/*.jpg"))
print(f'Total image: {len(images)}')

# index = 0
def image2percept(image_path : str = None):
    image_dir, image_name = os.path.split(image_path)
    image_stim = ImageStimulus(image_path)

    p2p_model = AxonMapModel()
    p2p_model.build()

    square16_implant = ProsthesisSystem(earray=ElectrodeGrid((16, 16), 300))
    square16_implant.stim = image_stim

    percept = p2p_model.predict_percept(square16_implant)

    saved_dir = image_dir.replace('patches', 'p2p')
    """Create an output directory if it doesn't exist"""
    Path(saved_dir).mkdir(parents=True, exist_ok=True)
    print(f'saving image to {saved_dir}')
    percept.plot()
    plt.savefig(saved_dir + '/' + image_name, bbox_inches='tight', pad_inches=0)
    plt.close()

for image in images:
    image2percept(image)
