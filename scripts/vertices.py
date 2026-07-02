#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:49:54 2018

@author: baillard

Package made to plot image, peform ginputs and handle vertices
 
"""

#
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import general.GMT as ggmt

#file_im='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/FIG/areas.png'
#img=mpimg.imread(file_im)
#plt.imshow(img)
#plt.grid()
#print('Pick two points for reference frame')
#ref=plt.ginput(n=2)
##print('Pick contour')
##contour=plt.ginput(n=0,timeout=0)
#
#ref_prime=[(4.76,1.83),(10.57,9)]

def ginput_vertices(image_file):
    img=mpimg.imread(image_file)
    fig,ax=plt.subplots()
    ax.imshow(img)
    ax.grid()
    ax.axis('equal')
    print('Pick two points for reference frame')
    ref=plt.ginput(n=2)
    print('Pick contour')
    vertices=plt.ginput(n=0,timeout=0)

    return (vertices,ref)

def fig2data_vertices(vertices,ref,ref_prime):
    if len(ref)!=2:
        raise ValueError('ref must have only 2 points, lower lef corner and upper corner')
        
    ### Process
    ll=ref[0]
    ur=ref[1]
    
    ll_prime=ref_prime[0]
    ur_prime=ref_prime[1]
    
    dx_prime=ur_prime[0]-ll_prime[0]
    dx=ur[0]-ll[0]
    dy_prime=ur_prime[1]-ll_prime[1]
    dy=ur[1]-ll[1]
    
    ### Convert
    
    x_prime=[(value-ll[0])*(dx_prime/dx)+ll_prime[0] for value,_ in vertices]
    y_prime=[(value-ll[1])*(dy_prime/dy)+ll_prime[1] for _,value in vertices]
    
    vertices_prime=list(zip(x_prime,y_prime))
    
    return vertices_prime

def read_vertices(vertices_file='vertices.dat'):
    vertex_list=[]
    with open(vertices_file,'rt') as fic:
        next(fic) # skip header
        for line in fic:
            ID=line.split(':')[0].strip()
            xy_list=[line.split(':')[1].split()[0],line.split(':')[1].split()[1] ]
            xy=[float(xy_list[0]),float(xy_list[1])]
            vertex_single=Vertex(ID=ID,xy=xy)
            vertex_list.append(vertex_single)
            
    Out=Vertices(vertices=vertex_list)
    return Out

def read_contours(contours_file='contours.dat',vertices_file='vertices.dat'):
    """
    """
    ### Read vertices file
    
    Cat_vert=read_vertices(vertices_file=vertices_file)
    
    ### Read contours files
    
    contour_list=[]
    with open(contours_file,'rt') as fic:
        next(fic) # skip header
        for line in fic:
            contour_id=line.split(':')[0].strip()
            line_vertex=line.split(':')[1].strip()
            vertex_ids=[vertex_id for vertex_id in line_vertex.split()]
            ### Make sure contours are closed
            if len(vertex_ids)<2:
                print('Number of vertices around contour must be >=2, skip')
                continue
            
            if vertex_ids[0] is not vertex_ids[-1]:

                vertex_ids.append(vertex_ids[0])

            contour_single=Cat_vert.select(vertex_ids=vertex_ids) 
            contour_single.ID=contour_id
            contour_list.append(contour_single)
            
    Out=Contours(contours=contour_list)
            
    return Out
 

#### Assign contour to vertices



    
class Vertex():
    """
    Vertex Class
    """
    def __init__(self,ID='',xy=None):
        self.ID=ID
        self.xy=xy
        
    def __str__(self):
        out='\''+self.ID+'\' :'
        if self.xy is not None:
            out+=str(self.xy[0])+'  '+str(self.xy[1])+'\n'
            
        return out
        
    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())
        
class Vertices():
    """
    Vertices class
    """
    
    def __init__(self,ID='',vertices=None):
        if not vertices:
            self.vertices = []
        else:
            self.vertices = vertices
            
        self.ID=ID
        
    def __getitem__(self, index):
        """
        __getitem__ method of the Catalog object to allow list accessibility
        :return: Event objects
        """

        if isinstance(index, slice):
            print(index)
            return self.__class__(vertices=self.vertices.__getitem__(index))
        else:
            return self.vertices.__getitem__(index)
        
    def __str__(self):
        out='%i vertices in object \'%s\' \n'%(len(self.vertices),self.ID)
        
        for vertex in self.vertices:
            out+=vertex.__str__()
            
        return out
    
    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())
        
    def select(self,vertex_ids=None):
        if vertex_ids is None:
            return Vertices()
        
        vertex_list=[x for vertex_id in vertex_ids for x in self.vertices if x.ID==vertex_id ]
        
        return Vertices(ID=self.ID,vertices=vertex_list)
    
    def get_xy(self):
        xy=[vertex.xy for vertex in self.vertices]
        
        return xy
    def write_vertices(self,vertices_file='vertices.dat'):
        """
        Write veritces dictionnary into file
        """
        with open(vertices_file,'wt') as fic:
            fic.write('Vertex ID (str) : X Y (floats)\n')
            for vertex_single in self.vertices:
                fic.write('%5s : %6.2f %6.2f\n'%(vertex_single.ID,vertex_single.xy[0],vertex_single.xy[1]))
    
    def plot_contour(self,ax=None,facecolor='r',text_color='w'):
        
        def get_bary(xy):
            x,y=list(zip(*xy))
            A=0
            Cx=0
            Cy=0
            for kk in range(len(xy)-1):
                A+=0.5*(x[kk]*y[kk+1]-x[kk+1]*y[kk])
                Cx+=(x[kk]+x[kk+1])*(x[kk]*y[kk+1]-x[kk+1]*y[kk])
                Cy+=(y[kk]+y[kk+1])*(x[kk]*y[kk+1]-x[kk+1]*y[kk])
                
            Cx=Cx/(6*A)
            Cy=Cy/(6*A)
            
            return (Cx,Cy)
        
    
        if ax is None:
            fig,ax=plt.subplots()

    
        xy=[vertex.xy for vertex in self.vertices]
        polygon = Polygon(xy,facecolor=facecolor,edgecolor='k')
        ax.add_patch(polygon)

        xy_bary=get_bary(xy)
    
        ax.text(xy_bary[0],xy_bary[1],self.ID,ha='center',va='center',weight='bold',color=text_color)
        
        return ax
        
class Contours():
    """
    Contours class
    """
    
    def __init__(self,ID='',contours=None):
        if not contours:
            self.contours = []
        else:
            self.contours = contours
            
        self.ID=ID
        
            
    def __getitem__(self, index):
        """
        __getitem__ method of the Catalog object to allow list accessibility
        :return: Event objects
        """

        if isinstance(index, slice):
            print(index)
            return self.__class__(contours=self.contours.__getitem__(index))
        else:
            return self.contours.__getitem__(index)
        
        
    def plot_contours(self,ax=None,facecolor='r',cmap=None):

        if ax is None:
            fig,ax=plt.subplots()
        
        if cmap is not None:
            facecolor=ggmt.data2rgb(range(len(self.contours)))
        else:
            if isinstance(facecolor,str):
                facecolor=[facecolor]*len(self.contours)
        for kk,contour in enumerate(self.contours):
            
            ax=contour.plot_contour(ax=ax,facecolor=facecolor[kk])
            
        return ax
    
    def select(self,contour_ids=None):
        if contour_ids is None:
            return Contours()
        
        contour_list=[x for contour_id in contour_ids for x in self.contours if x.ID==contour_id ]
        
        return Contours(ID=self.ID,contours=contour_list)

    def get_xy(self,ID):
        contour_select=self.select(contour_ids=[ID])
        
        xy=contour_select.contours[0].get_xy()
        
        return xy
    
    
        
        
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pickle


[x,y,z]=pickle.load(open('events.xyz','rb'))
plt.close('all')
#Cat=read_vertices()
contour_list=read_contours()
fig,ax=plt.subplots()

ax.plot(x,y,'ok',mfc='0.2',mec='none',ms=1,alpha=0.6)


contour_list.plot_contours(ax=ax,cmap=plt.cm.get_cmap('jet'))
A=contour_list.select(contour_ids='0')
xy=contour_list.get_xy(ID='0')







#colors = 100*np.random.rand(len(patches))
#
#
#p = PatchCollection(patches,alpha=0.4,cmap=plt.cm.get_cmap('jet'))
#p.set_array(np.array(colors))
#ax.add_collection(p)
ax.axis('equal')
ax.axis([2,12,0,10])



#
#vertices_prime=fig2data_vertices(contour,ref,ref_prime)
#
#fig,ax=plt.subplots()
#x,y=list(zip(*vertices_prime))
#ax.plot(x,y,'ok')
#ax.axis('equal')
#
#### Give labels to vertices
#
#### Reorder
#x=np.asarray(x)
#y=np.asarray(y)
#
#data=np.column_stack((x,y))
#
#data_sort=data[np.argsort(-data[:,1])]
#
#dic_vertices={}
#
#vertex_list=[]
#for kk in range(data_sort.shape[0]):
#    dic_vertices[str(kk)]=list(data_sort[kk,:])
#    vertex_list.append(Vertex(name=str(kk),xy=list(data_sort[kk,:])))
#    
#Cat=Vertices(vertices=vertex_list)
#Cat.write_vertices()

#fig,ax=plt.subplots()
#for key, value in dic_vertices.items():
#    ax.plot(value[0],value[1],'ok')
#    ax.text(value[0],value[1],key)
#    
#ax.axis('equal')
#
#### Write vertex file
#

        
        
            
    
