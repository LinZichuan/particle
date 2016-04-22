require 'torch'

--local cmd = 'ls ~/Desktop/qtcreator_5.0.2/build-test1-Desktop_Qt_5_0_2_GCC_64bit-Debug/ | grep "^star_\\|^noise_"'
local cmd = 'ls /home/lzc/particle/test1/ | grep "bin"'
local handle = io.popen(cmd)
local res = handle:read("*a")
handle:close()
local ar = {}
for i in string.gmatch(res, "%S+") do
    ar[#ar+1] = i
end
print (ar)
local num = #ar

--x86 is Little endian
function bytes_to_int(b1, b2, b3, b4)
      if not b4 then error("need four bytes to convert to int",2) end
      local n = b1 + b2*256 + b3*65536 + b4*16777216
      n = (n > 2147483647) and (n - 4294967296) or n
      return n
end

number = arg[1]
for i=1,1 do
    local binfile = 'split_image_stack_'..number..'_cor.mrc.bin' --'split_image.bin'--ar[i]
    local base = '/home/lzc/particle/all_split_image/'--'/home/lzc/particle/test1/'
    local inp = assert(io.open(base..binfile, 'rb'))

    local data = {}
    while true do
        local bytes = inp:read(4)
        if bytes == nil then break end
        local st = bytes_to_int(bytes:byte(1,4))
        data[#data+1] = st
    end
    inp:close()

    local tdata = torch.Tensor(data)
    --local vdata = tdata:view(36,88,88)
    print(tdata:storage():size())
    local vdata = tdata:view(tdata:storage():size()/10000, 100, 100)

    print('saving into ' .. binfile .. '.t7......')
    torch.save('./all_split_image/' .. binfile .. '.t7', vdata)
end