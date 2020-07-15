from CountSketch import * 
import time

d = 50 * 10**2; c = 10**3; r = 5; k = 10

g1 = torch.randint(0, 100000, (d,), dtype=torch.int64, device="cuda")
g2 = torch.ones(d, dtype=torch.int64, device="cuda")
g2[:d//2] = 324 
g2[d//2:] = 327 

csv2 = CSVec( d, c, r, k)
csv2.accumulateVec(g1)
csv2.accumulateVec(g2)

ind =  torch.ones(3, dtype=torch.int64, device='cuda')
ind[0] = 323
ind[1] = 324
ind[2] = 327
print (csv2.findValues(ind))
print (csv2.getTopk())
#time.sleep(5)

