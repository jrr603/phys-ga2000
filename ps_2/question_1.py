#### PROBLEM 1 on PS2 

import numpy as np
import matplotlib as plt

def get_bits(number):
    bytes = number.tobytes()
    bits = []
    for byte in bytes:
        bits = bits + np.flip(np.unpackbits(np.uint8(byte)), np.uint8(0)).tolist()
    return list(reversed(bits))
mynumber= 100.98763
for value in [mynumber]:
    bitlist=get_bits(np.float32(value))
    sign = bitlist[0]
    exponent = bitlist[1:9]
    mantissa = bitlist[9:32]
template = """\n{value} in NumPy’s 32-bit floating point representation is:
   sign = {sign}
   xponent = {exponent}
   mantissa = {mantissa}"""
print(template.format(value=value, sign=sign, exponent=exponent, mantissa=mantissa))

def get_int(mantissa_):
    ssum =0
    for i in range(23):
     ssum += ( mantissa_[22-i]* pow(2,(i-23)))
     #print(mantissa_[22-i])
    theint = pow(2,(133-127))*(1+ssum)
    return(theint)

diff = get_int(mantissa) - mynumber
print('These Bits actually represent: ',get_int(mantissa),'\n\nso lets subtract them and find their difference: \n', diff)