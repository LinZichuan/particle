
traindata = torch.load('./sample/2traindata.t7')
testdata = torch.load('./sample/2testdata.t7')

size = traindata:storage():size()/10000
print (size)

module = nn.SpatialAveragePooling(100, 100, 2, 2)

for i = 1,size do
    traindata[i]
end
