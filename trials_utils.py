import pickle

if __name__ == "__main__":
    fn = 'log/31-01-2021-21:41:08-opt.pickle'  #change filename
    fh = open(fn, 'rb')
    a = pickle.load(fh)

    for i in range(len(a[0])):
        print("Score: ", a[0][i], " Param: ", a[1][i])