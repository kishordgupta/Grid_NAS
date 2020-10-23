import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import matplotlib.image as mpimg
import numpy as np
 
 
def NSgridtrain(selfdata,feature_value_count=24,encodesize=8,xorstyle=1,gridsize=4,grid=1024,maxcoverage=1,mincoverage=1, gridcoverage=500,visualize=False):
  max=selfdata.max()+(selfdata.std()*maxcoverage)
  min=selfdata.min()+(selfdata.std()*mincoverage)
  selfdata=normal(selfdata,max,min)
  #print(selfdata)
  arr=selfdata_encode(selfdata,encodesize,feature_value_count)
  #print(arr)
  res=selfdataxor(arr,xorstyle,gridsize)
  res1=addcoverage(res,gridcoverage,visualize)
  if visualize==True:
    fig,ax=plt.subplots(1,2,figsize=(20,10))
    griddraw(res,grid)
    im1=mpimg.imread('imagedraw.jpg')
    griddraw(res1,grid)
    im2=mpimg.imread('imagedraw.jpg')
    # fig.add_subplot(2, 1, 1)
    ax[0].imshow(im1)
    # fig.add_subplot(2, 1, 2)
    ax[1].imshow(im2)
    plt.show()
  return res1,min,max
,
def normal(df,max,min):
  norm = (df - min) / (max -min )
  return norm
 
def printgrid(arr,datarownumber,feature_value_count):
  #print(arr)
  s=''
  for rowid in range(datarownumber):
    for index in range(feature_value_count):
      s=s+str(arr[index][rowid])
    s=s+"\n"
  return s
 
def makehalf(line,length):
  s=""
  line = line.replace('\n','')
  for i in range(length):
    k=int(line[i])+int(line[length+i])
    if k==1:
      s=s+'1'
    else:
      s=s+'0'
  return s
 
 
def selfdata_encode(data,encodesize,feature_value_count):
  partition = float(100/encodesize)
  datarownumber = len(data.index)
  arr=[[0 for row in range(0,datarownumber)] for col in range(0,feature_value_count)]
  for rowid in range(feature_value_count):
    s=''
    for index, row in df.iterrows():
        i=row[rowid]
        i=int((i*100)/partition)
        s=s+str(i)
        arr[rowid][index]=str('{0:03b}'.format(i))
  arr=printgrid(arr,datarownumber,feature_value_count)
  return arr
  
def selfdataxor(arr,xorstyle,gridsize):
  Lines = arr.split("\n")
  x=[]
  i=1
  for s in Lines:
    #print(line)
    i=1
    #s=makehalf(s,int(len(s)/2))
    while 1:
      s=makehalf(s,int(len(s)/2))
      i=i+1
      if i>gridsize:
        break
    #print(s)
    if len(s)>1:
      x.append(int(s,2))
  res = sorted(set(x), key = lambda ele: x.count(ele)) 
  res=sorted(res)
  return res
 
def gridmaker(res,grid):
  n=grid
  matrix=[['0' for row in range(0,n)] for col in range(0,n)]
  for b in res:
    if b<n*n:
      matrix[int(b/n)][int(b%n)]="X"    
  for l in range(n):
    s=""
    for k in range(n):
      s=s+matrix[l][k]
    print(s)
 
def griddraw(res,grid):
  n=grid
  matrix=[['0' for row in range(0,n)] for col in range(0,n)]
  for b in res:
    if b<n*n:
      matrix[int(b/n)][int(b%n)]="X"
  drawgrid(matrix,n,s=6)    
  return matrix
 
def drawgrid(matrix,n,s=3):
  im = Image.new('RGB', (n*s, n*s), (128, 128, 128))
  draw = ImageDraw.Draw(im)
  for l in range(n):
    for k in range(n):
      if matrix[k][l]=='X':
        draw.rectangle((l*s, k*s, (l+1)*s, (k+1)*s), fill=(255, 0, 0), outline=(255, 255, 255))
      else:
        draw.rectangle((l*s, k*s,  (l+1)*s, (k+1)*s), fill=(0, 255, 0), outline=(255, 255, 255))
      #k=k+s
    #l=l+s
    #im.show()
  im.save('imagedraw.jpg', quality=95)
  return im
  
import numpy as np
def addcoverage(griddata,coverage,visualize):
  # print(griddata)
  data =  np.zeros((len(griddata),2*coverage+1))
  data[:,coverage] = np.array(griddata).reshape(-1)
  for k in range(coverage):
    data[:,coverage-k-1] = data[:,coverage] - k-1
    data[:,coverage+k+1] = data[:,coverage] + k+1
  data = data.reshape(-1)
  data = data[data>=0]
  data = sorted(np.unique(data))
  # print(data)


  # for a in griddata:
  #   for i in range(coverage):
  #     if a-i>0:finaldata.append(a-i)
  #     finaldata.append(a+i)
  # res = sorted(set(finaldata), key = lambda ele: finaldata.count(ele)) 
  # res=sorted(res)
  return data
 
 
def NSgridtest(griddata,nonselfdata,max,min,feature_value_count=24,encodesize=8,xorstyle=1,gridsize=4,grid=1024,maxcoverage=1,mincoverage=1, gridcoverage=1,visualize=False):
  nonelfdata=normal(nonselfdata,max,min)
  #print(nonelfdata)
  arr=selfdata_encode(nonelfdata,encodesize,feature_value_count)
  #print(arr)
  res=selfdataxor(arr,xorstyle,gridsize)
  acc=common_member(griddata,res,gridcoverage)/len(res)
  print("accuracy:"+str(acc))
  if visualize==True:
    griddraw(res,grid)
  return res
 
def common_member(a, b,add): 
    a_set = set(a) 
    b_set = set(b) 
  
    if (a_set & b_set): 
        print(a_set & b_set)
        return len(a_set & b_set) 
    else: 
        print("No common elements")
    return 0
