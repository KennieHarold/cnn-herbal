import splitfolders

input_folder = "raw_images/"
splitfolders.ratio(input_folder, output="./", seed=1337, ratio=(.8, .0, .2), group_prefix=None, move=False)
