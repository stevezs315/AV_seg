#--------data preprocessing-----------------#
# training
# python3 preprocessing.py --images-path train/_Data/GAVE_preliminary/GAVE/training/images \
# --masks-path train/_Data/GAVE_preliminary/GAVE/training/masks --save-path train/_Data/GAVE_preliminary/GAVE/training/enhanced

# test
# python3 preprocessing.py --images-path train/_Data/GAVE_preliminary/GAVE/GAVE_test/images \
# --masks-path train/_Data/GAVE_preliminary/GAVE/GAVE_test/masks --save-path train/_Data/GAVE_preliminary/GAVE/GAVE_test/enhanced


# python3 preprocessing.py --images-path train/_Data/RITE_original/RITE/AV_groundTruth/test/images_blurred \
# --masks-path train/_Data/RITE/test/enhanced_masks --save-path train/_Data/RITE_original/RITE/AV_groundTruth/test/images_blurred_enhanced

# large-field private preprocessing
python3 preprocessing.py --images-path train/_Data/large_field_private/training/images \
--masks-path train/_Data/large_field_private/training/masks --save-path train/_Data/large_field_private/training/enhanced

# python3 preprocessing.py --images-path train/_Data/large_field_private/validation/images \
# --masks-path train/_Data/large_field_private/validation/masks --save-path train/_Data/large_field_private/validation/enhanced
