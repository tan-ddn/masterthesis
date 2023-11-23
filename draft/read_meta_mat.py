import scipy.io
label_array = scipy.io.loadmat('masterthesis/meta.mat')['synsets']

label_dic = {}
for i in  range(1000):
    label_dic[label_array[i][0][1][0]] = i

print(label_dic)
