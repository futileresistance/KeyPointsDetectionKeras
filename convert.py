import coremltools
from test import thin_test_model

ios_model_name = 'ios_models/LIGHT_thin_model_IOS.mlmodel'
print('converting...')
ios_model_mobilenet = coremltools.converters.keras.convert(thin_test_model,\
        input_names='image', image_input_names="image", output_names='heatmaps', image_scale=1/255.0)
print('saving...')
ios_model_mobilenet.save(ios_model_name)
print('done!')