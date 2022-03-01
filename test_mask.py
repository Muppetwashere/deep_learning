from src.rollwear_lib_object.models.strips import Mask
from src.rollwear_lib_object.wearcentre_from_strips import train_neuralnet
mask = Mask(True, True, False, False, True, True, True, True, True, False)

train_neuralnet(250, (20, 8), ('selu', 'selu'), 'test',mask=mask, recurrent=False, f6=True, top=True)