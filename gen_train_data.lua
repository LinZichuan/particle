require 'torch'

local cmd = 'ls ~/Desktop/qtcreator_5.0.2/build-test1-Desktop_Qt_5_0_2_GCC_64bit-Debug/ | grep "^star_\\|^noise_"'
local handle = io.popen(cmd)
local res = handle:read("*a")
handle:close()
local ar = {}
for i in string.gmatch(res, "%S+") do
    ar[#ar+1] = i
end
print (ar)
local num = #ar

local star = {}
local noise = {}
--x86 is Little endian
function bytes_to_int(b1, b2, b3, b4)
    if not b4 then error("need four bytes to convert to int",2) end
    local n = b1 + b2*256 + b3*65536 + b4*16777216
    n = (n > 2147483647) and (n - 4294967296) or n
    return n
end
local byte
local st
for i=1,num do
    local binfile = ar[i]
    print(binfile)
    local base = '/home/linzichuan/Desktop/qtcreator_5.0.2/build-test1-Desktop_Qt_5_0_2_GCC_64bit-Debug/'
    local inp = assert(io.open(base..binfile, 'rb'))

    if (string.find(ar[i], 'star')) then
        while true do
            bytes = inp:read(4)
            if bytes == nil then break end
            st = bytes_to_int(bytes:byte(1,4))
            star[#star+1] = st
        end
    end
    if (string.find(ar[i], 'noise')) then
        while true do
            bytes = inp:read(4)
            if bytes == nil then break end
            st = bytes_to_int(bytes:byte(1,4))
            noise[#noise+1] = st
        end
    end

    inp:close()
    collectgarbage()

    print (#star*4/1024/1024 .. 'MB')
    print (#noise*4/1024/1024 .. 'MB')
end
print('start converting,..')
stardata = torch.Tensor(star)
stardata = stardata:add(-stardata:mean())
stardata = stardata:div(stardata:std())
starimagesize = stardata:storage():size()/10000
stardata = stardata:view(starimagesize, 100, 100)

noisedata = torch.Tensor(noise)
noisedata = noisedata:add(-noisedata:mean())
noisedata = noisedata:div(noisedata:std())
noiseimagesize = noisedata:storage():size()/10000
noisedata = noisedata:view(noiseimagesize, 100, 100)

print('saving into ./sample/star.t7......')
torch.save('./sample/star.t7', stardata)
print(stardata:size())

print('saving into ./sample/noise.t7......')
torch.save('./sample/noise.t7', noisedata)
print(noisedata:size())

traindata = {}
label = {}
for i=1,starimagesize do
    label[i] = '1'
end
for i=starimagesize+1,starimagesize+noiseimagesize do
    label[i] = '2'
end
traindata['y'] = torch.Tensor(label)--torch.cat(torch.ones(starimagesize), torch.zeros(noiseimagesize))
all = torch.cat(torch.Tensor(star), torch.Tensor(noise))
traindata['X'] = all:view(all:storage():size()/10000, 100, 100)

--shuffle order
local totalimagesize = all:storage():size() / 10000
local randindices = torch.randperm(totalimagesize)
local shuffleX = torch.Tensor(totalimagesize, 100, 100)
local shuffley = torch.IntTensor(totalimagesize)
for i=1,totalimagesize do
    shuffleX[i] = traindata['X'][randindices[i]]:clone()
    shuffley[i] = traindata['y'][randindices[i]]
end
traindata['X'] = shuffleX:clone()
traindata['y'] = shuffley:clone()
print(traindata['y']:dim())
print(traindata['X']:dim())

for i=1,totalimagesize do
    assert(traindata['y'][i] == 1 or traindata['y'][i] == 2)
end
print('saving into ./sample/train_2506x100x100.t7......')
torch.save('./sample/train_2506x100x100.t7', traindata)


