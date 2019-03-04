
fname = 'yoloLayers.txt'
with open(fname) as f:
    lines = f.readlines()
lines = [line.strip() for line in lines if line[0]!='#']
sizew = 0
sizefloat = 4
sizeint = 4
for line in lines:
    linesp = line.split(' ')
    if linesp[1] == 'conv':
        n = int(linesp[2])
        size = int(linesp[3].split('x')[0])
        c = int(linesp[4].split('x')[-1])
        bias = sizefloat*n
        scale = sizefloat*n
        mean = sizefloat*n
        variance = sizefloat*n
        weights = sizefloat*n*c*size*size
        sizew += (bias+scale+mean+variance+weights)
    if linesp[1] == 'Local':
        n = 256
        size = 3
        c = 1024
        outwh = 7
        bias = sizefloat*outwh*outwh*n
        weights = sizefloat*size*size*c*n*outwh*outwh
        sizew += (bias+weights)
    if linesp[1] == 'connected':
        outputs = 7*7*35
        inputs = 7*7*256
        bias = sizefloat*outputs
        weights = sizefloat*outputs*inputs
        sizew += (bias+weights)

print('batchnormalized convolutional weight size:%d bytes'%(sizew+sizeint*4))
#conv20:89721616
#conv24:240782096
#conv+Local:703254288
#conv+Local+connected:789312988
