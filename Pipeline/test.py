import argparse

global folder

def folder_name(directory):
    folder = directory
    print("Inside folder")
    print(folder)

print("Global variable")
#print(folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enter the directory name where you have the test data')
    parser.add_argument('--dir', default='test_images', help='default directory name for the test images')
    args = parser.parse_args()

    print("INSIDE MAIN")

    folder_name(args.dir)
