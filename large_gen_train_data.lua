require 'torch'

local basename = 'hzhoubin/'
--local cmd = 'ls ./gammasbin | grep "\\.bin"'
local cmd = 'ls ./'..basename..' | grep "\\.bin"'
local handle = io.popen(cmd)
local res = handle:read("*a")
handle:close()
local ar = {}
for i in string.gmatch(res, "%S+") do
    ar[#ar+1] = i
end
print (ar)
local num = #ar

local side = 80
local size = side * side
--x86 is Little endian
function bytes_to_int(b1, b2, b3, b4)
    if not b4 then error("need four bytes to convert to int",2) end
    local n = b1 + b2*256 + b3*65536 + b4*16777216
    n = (n > 2147483647) and (n - 4294967296) or n
    return n
end

for i=1,num do
    local binfile = ar[i]
    print(binfile)
    local inp = assert(io.open(basename..binfile, 'rb'))

    if (string.find(ar[i], 'star')) then
        local star = {}
        while true do
            local bytes = inp:read(4)
            if bytes == nil then break end
            local st = bytes_to_int(bytes:byte(1,4))
            star[#star+1] = st
        end
        if (#star > 0) then
            if traindata == nil then
                traindata = torch.Tensor(star)
            else
                traindata = traindata:cat(torch.Tensor(star))
            end
            print(#star/size)
            if trainlabel == nil then
                trainlabel = torch.Tensor(#star/size):fill(2)
            else 
                trainlabel = trainlabel:cat(torch.Tensor(#star/size):fill(1))
            end
            star = nil
        end
    end
    if (string.find(ar[i], 'noise')) then
        local noise = {}
        while true do
            local bytes = inp:read(4)
            if bytes == nil then break end
            local st = bytes_to_int(bytes:byte(1,4))
            noise[#noise+1] = st
        end
        if (#noise > 0) then
            if traindata == nil then
                traindata = torch.Tensor(noise)
            else
                traindata = traindata:cat(torch.Tensor(noise))
            end
            print(#noise/size)
            if trainlabel == nil then
                trainlabel = torch.Tensor(#noise/size):fill(2)
            else
                trainlabel = trainlabel:cat(torch.Tensor(#noise/size):fill(2))
            end
            noise = nil
        end
    end

    inp:close()
    collectgarbage()
end
print('total image size = ' .. trainlabel:storage():size())
print('start shuffling...')

--traindata['y'] = torch.Tensor(label)--torch.cat(torch.ones(starimagesize), torch.zeros(noiseimagesize))
--all = torch.cat(torch.Tensor(star), torch.Tensor(noise))
--traindata['X'] = all:view(all:storage():size()/10000, 100, 100)
--
----shuffle order
local totalsize = trainlabel:storage():size()
local randindices = torch.randperm(totalsize)
traindata = traindata:view(totalsize, side, side)
shuffley_ = {trainlabel[randindices[1]]}
local batch = 200
local t = {}
for k=1,side do
    for j=1,side do
        t[#t+1] = traindata[randindices[1]][k][j]
    end
end
shuffleX = torch.Tensor(t)
t = {}
for i=2, totalsize do
    xlua.progress(i, totalsize)
    for k=1,side do
        for j=1,side do
            t[#t+1] = traindata[randindices[i]][k][j]
        end
    end
    if i % batch == 0 then
        shuffleX = shuffleX:cat(torch.Tensor(t))
        t = {}
    end
    shuffley_[i] = trainlabel[randindices[i]]
end
shuffleX = shuffleX:cat(torch.Tensor(t))
shuffleX = shuffleX:view(shuffleX:storage():size()/size, side, side)
shuffley = torch.Tensor(shuffley_)
print (shuffleX:size())
print (shuffley:size())


local trainsize = math.floor(totalsize * 0.9)
local testsize = totalsize - trainsize

train_data  = torch.Tensor(trainsize, side, side)
test_data   = torch.Tensor(testsize, side, side)
train_label = torch.Tensor(trainsize)
test_label  = torch.Tensor(testsize)

local struct = require('struct')
--local inptr = assert(io.open('sample/total-train-data.bin', 'wb'))
--local inpte = assert(io.open('sample/total-test-data.bin', 'wb'))
for i=1,trainsize do
    train_data[i]  = shuffleX[i]
    train_label[i] = shuffley[i]
    --[[for k=1,side do
        for l=1,side do
            inptr:write(struct.pack('i4', train_data[{i,k,l}]))
        end
    end]]
end

for i=1,testsize do
    test_data[i]   = shuffleX[i+trainsize]
    test_label[i]  = shuffley[i+trainsize]
    --[[for k=1,side do
        for l=1,side do
            inpte:write(struct.pack('i4', test_data[{i,k,l}]))
        end
    end]]
end
--assert(inptr:close())
--assert(inpte:close())

--torch.save('./sample/gammas_traindata.t7', train_data)
--torch.save('./sample/gammas_trainlabel.t7', train_label)
--torch.save('./sample/gammas_testdata.t7', test_data)
--torch.save('./sample/gammas_testlabel.t7', test_label)


torch.save('./sample/hzhou_traindata_manual.t7', train_data)
torch.save('./sample/hzhou_trainlabel_manual.t7', train_label)
torch.save('./sample/hzhou_testdata_manual.t7', test_data)
torch.save('./sample/hzhou_testlabel_manual.t7', test_label)

