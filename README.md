# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# How to run
Make sure to navigate to the rootdir of the files provided:

/aipnd-project>

# Running The train.py file
Run:  python train.py flowers --save_dir save_directory --arch "vgg13" --learning_rate 0.01 --hidden_units 512 --epochs 20 --gpu

# Running The predict.py file

Run: python predict.py --top_k 3 --category_names cat_to_name.json --gpu ./flowers/test/19/image_06186.jpg ./save_dir/checkpoint.pth

