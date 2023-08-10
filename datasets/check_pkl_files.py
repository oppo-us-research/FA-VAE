"""
* Copyright (c) 2023 OPPO. All rights reserved.
* SPDX-License-Identifier: MIT
* For full license text, see LICENSE.txt file in the repo root
"""

"""
This script is to show first two rows in the pkl files
"""

import pickle as pk
import glob

def main():
    data_dir = "/nfs/home/uss00087/fa-vae/datasets/pkl_files/*"
    pkl_files_lst = glob.glob(data_dir)
    print("\npkl_files = ", pkl_files_lst)

    for pkl_file in pkl_files_lst:
        with open(pkl_file, "rb") as input_file:
            names = pk.load(input_file)
        length = len(names)
        keys = names.keys()
        # print(names[0])
        print("\n=========== for file {}, length {}".format(pkl_file, length))

        i = 0
        for key, value in names.items():
            print("an example {} {}".format(key, value))
            i += 1
            if i > 1:
                break

if __name__ == "__main__":
    main()