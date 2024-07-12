import os

# train test split percentage -> 80% train, 20% test
dd = os.listdir("dataset")
dd.remove('.DS_Store')

folds = ['folds/fold_1.txt', 'folds/fold_2.txt', 'folds/fold_3.txt', 'folds/fold_4.txt', 'folds/fold_5.txt']

for i in range(len(dd)):
    d2 = os.listdir("dataset/%s/" % (dd[i]))
    d2.remove('.DS_Store')
    for j in range(len(folds)):
        if i == 0:
            f = open(folds[j], 'w')
        else:
            f = open(folds[j], 'a')
            
        for k in range(j * 10, j * 10 + 10):
            str = "dataset/%s/%s" % (dd[i], d2[k])
            f.write("%s %d\n" % (str, i)) 
        f.close()
            
for i in range(len(folds)):
    # testing data
    with open('folds/fold_%d.txt' % (i+1), 'r') as input_file:
        content = input_file.read()
    with open('train_test_data/test_%d.txt' % (i+1), 'w') as output_file:
        output_file.write(content)
        
    # training data
    n = 0
    for file in folds:
        if file != 'folds/fold_%d.txt' % (i+1):
            with open(file, "r") as input_file:
                content = input_file.read()
            if n == 0:
                with open('train_test_data/train_%d.txt' % (i+1), 'w') as output_file:
                    output_file.write(content)
            else:
                with open('train_test_data/train_%d.txt' % (i+1), 'a') as output_file:
                    output_file.write(content)
            n += 1