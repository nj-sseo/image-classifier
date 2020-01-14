import os
import shutil
import numpy as np



def split_data(ver, p = 0.2):
    os.system('unzip train_ens -d train_ens')

    cv_dir = './cv{}/'.format(ver)
    os.makedirs(cv_dir)

    source1 = './train_ens/'
    dest1 = cv_dir + 'train/'
    dest2 = cv_dir + 'valid/'
    dirs = os.listdir(source1)

    for files in dirs:
        for f in os.listdir(source1 + files):
            dest_dir1 = dest1 + files
            dest_dir2 = dest2 + files
            print(dest_dir1, files)
            if not os.path.isdir(dest_dir1): os.makedirs(dest_dir1, exist_ok = True)
            if not os.path.isdir(dest_dir2): os.makedirs(dest_dir2, exist_ok = True)

            if np.random.rand(1) > p:
                shutil.move("{0}{1}/{2}".format(source1, files, f), dest_dir1)
            else:
                shutil.move("{0}{1}/{2}".format(source1, files, f), dest_dir2)
    print("Done~")
    os.system('rm .DS_Store')
    os.system('rm -r __MACOSX')
    return [dest1, dest2]

def count_data(data_dir):
    train_dir = os.path.abspath(data_dir)
    dirs = os.listdir(train_dir)
    with open("count_{}.csv".format(data_dir), 'w') as f:
        for files in dirs:
            print(files, len(os.listdir(train_dir+"/"+files)))
            f.write("%s,%s\n"%(files, str(len(os.listdir(train_dir+"/"+files)))))


def over_sampling(data_dir = './train_over'):
    train_dir = os.path.abspath(data_dir)
    dirs = os.listdir(train_dir)
    for files in dirs:
        file_id = 0
        file_list = os.listdir('{}/{}'.format(data_dir, files))
        #print(file_list)
        while len(os.listdir('{}/{}'.format(data_dir, files))) < 300:
            curr_dir = '{}/{}/'.format(data_dir, files)
            random_file = np.random.choice(file_list, 1)
            print(curr_dir + random_file[0])
            os.system('cp {} {}'.format( curr_dir + random_file[0], curr_dir + str(file_id) + '.jpg'))
            file_id += 1
