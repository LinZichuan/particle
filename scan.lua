require 'torch'
require 'nn'
require 'optim'
require 'cutorch'
require 'cunn'

classes = {'1', '2'}
local opt = lapp[[
    -n, --network   (default "network_cuda.t7")    reload pretrained network
    -s, --step      (default '100')                    step of scanning the image
]]
print('<trainer> reloading previously trained network')
model = torch.load(opt.network)
print(model)
criterion = nn.ClassNLLCriterion()
confusion = optim.ConfusionMatrix(classes)

function scan(patchdata, size)
    local res = {}
    local total = 0
    local last = 0
    for i=1, size, 64 do
        xlua.progress(i, size)
        last = i + 63
        if last > size then
            last = size
        end 
        local bs = last - i + 1
        local inputs = torch.CudaTensor(bs, 1, 100, 100)
        for j=1,bs do
            inputs[{j,1}] = patchdata[i+j-1]
        end 
        local outputs = model:forward(inputs)
        for j=1,bs do
            print (outputs[j][1] .. ', ' .. outputs[j][2])
            if outputs[j][1] > outputs[j][2] then
                print (i+j-1)
                total = total + 1
                res[#res+1] = i+j-1
            end
        end
    end
    print ('total number is ' .. total)
    return res
end

print  (arg[1])
number = arg[1] --'0003'
local splitimage = torch.load('all_split_image/split_image_stack_'..number..'_cor.mrc.bin.t7'):cuda() --torch.load('sample/split_image.bin.t7'):cuda()
local size = splitimage:storage():size()/10000
print (size)
local res = scan(splitimage, size)
print(res)

trueimage = torch.Tensor(#res, 100, 100)
local inp = assert(io.open('all_split_image/scanres_stack_'..number..'.bin', 'wb'))
local struct = require('struct')

local rr = 3710
local cc = 3838
local step = 50
local col = math.floor(cc/50)
for i=1,#res do
    trueimage[i] = splitimage[res[i]]:float()
    local indexr = math.floor((res[i]-1) / col) * step
    local indexc = math.floor((res[i]-1) % col) * step
    print(indexr .. ',' .. indexc)
    inp:write(struct.pack('i4', indexr))
    inp:write(struct.pack('i4', indexc))
end

assert(inp:close())

local fn = 'all_split_image/trueimage'..number..'.t7'
torch.save(fn, trueimage)
os.execute('scp '..fn..' linzichuan@166.111.131.163:~/Study/senior_second/particle/show/trueimage.t7')
os.execute('scp all_split_image/scanres_stack_'..number..'.bin' .. ' linzichuan@166.111.131.163:~/Desktop/qtcreator_5.0.2/build-test1-Desktop_Qt_5_0_2_GCC_64bit-Debug/scanres')
