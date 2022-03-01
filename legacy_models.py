import src.rollwear_lib_object.wearcentre_from_strips
import src.rollwear_lib_object.models.abstract_models as models
mask = models.Mask(f6=True, top=True,fw_and_l=True,fwl=True,strips_param=True,roll_param=True,hardness_indic=True,family=True,suppliers=True,cum_length=True)
src.rollwear_lib_object.wearcentre_from_strips.train_neuralnet(250, (20, 8), ('selu', 'selu', 'sigmoid'), 
                                                       'test',mask=mask, recurrent=False, f6=True, top=True)
                                                       