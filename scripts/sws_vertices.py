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
import GMT as ggmt
        
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pickle
from matplotlib.widgets import Cursor,TextBox


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

def zoom_factory(ax,base_scale = 2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        ax.set_ylim([ydata - cur_yrange*scale_factor,
                     ydata + cur_yrange*scale_factor])
        plt.draw() # force re-draw

    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)

    #return the function
    return zoom_fun

def ginput_vertices(image_file):
    img=mpimg.imread(image_file)
    fig,ax=plt.subplots(figsize=[9,9])

    ax.imshow(img)
    ax.set_aspect('equal')
    ### Make sure frame fits to data
    ax.autoscale(enable=True, axis='both', tight=True)
    fig.tight_layout()
    window = plt.get_current_fig_manager().window
    geom = window.geometry()
    x,y,dx,dy = geom.getRect()
    window.setGeometry(1,1,dx, dy)
    
    f = zoom_factory(ax, base_scale=1.1)
    cursor = Cursor(ax, useblit=True, color='k', linewidth=1)
    
    print('Pick two points for reference frame')
    ref=plt.ginput(n=2)
    
    continue_pick=True
    
    vertices_list=[]
    while continue_pick:
        print('Pick contour')    
        vertices=plt.ginput(n=0,timeout=0)
        vertices_list.append(vertices)
        str_cmd='Do you want to continue [y/n]?'
        answer=input(str_cmd)
        if answer=='y':
            continue_pick=True
        else:
            continue_pick=False
            
    return (vertices_list,ref)


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
            key=line.split(':')[0].strip()
            xy_list=[line.split(':')[1].split()[0],line.split(':')[1].split()[1] ]
            xy=[float(xy_list[0]),float(xy_list[1])]
            vertex_single=Vertex(key=key,xy=xy)
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
            contour_key=line.split(':')[0].strip()
            line_vertex=line.split(':')[1].strip()
            vertex_keys=[vertex_key for vertex_key in line_vertex.split()]
            ### Make sure contours are closed
            if len(vertex_keys)<2:
                print('Number of vertices around contour must be >=2, skip')
                continue
            
            if vertex_keys[0] is not vertex_keys[-1]:

                vertex_keys.append(vertex_keys[0])

            contour_single=Cat_vert.select(vertex_keys=vertex_keys) 
            contour_single.key=contour_key
            contour_list.append(contour_single)
            
    Out=Contours(contours=contour_list)
            
    return Out
 

#### Assign contour to vertices



    
class Vertex():
    """
    Vertex Class
    """
    def __init__(self,key='',xy=None):
        self.key=key
        self.xy=xy
        
    def __str__(self):
        out='\''+self.key+'\' :'
        if self.xy is not None:
            out+=str(self.xy[0])+'  '+str(self.xy[1])+'\n'
            
        return out
        
    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())
        
class Vertices():
    """
    Vertices class
    """
    
    def __init__(self,key='',vertices=None):
        if not vertices:
            self.vertices = []
        else:
            self.vertices = vertices
            
        self.key=key
        
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
        out='%i vertices in object \'%s\' \n'%(len(self.vertices),self.key)
        
        for vertex in self.vertices:
            out+=vertex.__str__()
            
        return out
    
    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())
        
    def select(self,vertex_keys=None):
        if vertex_keys is None:
            return Vertices()
        
        vertex_list=[x for vertex_key in vertex_keys for x in self.vertices if x.key==vertex_key ]
        
        return Vertices(key=self.key,vertices=vertex_list)
    
    def get_xy(self):
        xy=[vertex.xy for vertex in self.vertices]
        
        return xy
    def write_vertices(self,vertices_file='vertices.dat'):
        """
        Write veritces dictionnary into file
        """
        with open(vertices_file,'wt') as fic:
            fic.write('Vertex key (str) : X Y (floats)\n')
            for vertex_single in self.vertices:
                fic.write('%5s : %6.2f %6.2f\n'%(vertex_single.key,vertex_single.xy[0],vertex_single.xy[1]))
    
    def plot_contour(self,ax=None,facecolor='r',textcolor='w'):
        
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
    
        ax.text(xy_bary[0],xy_bary[1],self.key,ha='center',va='center',weight='bold',color=textcolor)
        
        return ax
        
class Contours():
    """
    Contours class
    """
    
    def __init__(self,key='',contours=None):
        if not contours:
            self.contours = []
        else:
            self.contours = contours
            
        self.key=key
        
            
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
        
    def __str__(self):
        out='%i contours in object \'%s\' \n'%(len(self.contours),self.key)
        
        for contour in self.contours:
            out+=contour.__str__()
            
        return out
    
    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())
        
    def get_contour_keys(self):
        
        contour_keys=[x.key for x in self.contours]
        return contour_keys
        
        
    def plot_contours(self,ax=None,facecolor='r',cmap=None,contour_keys=None,textcolor='w'):

        if contour_keys is None:
            contour_keys=self.get_contour_keys()
            
        sel_contour=self.select(contour_keys=contour_keys)

            
        if ax is None:
            fig,ax=plt.subplots()
        
        if cmap is not None:
            facecolor=ggmt.data2rgb(range(len(sel_contour.contours)),cmap=cmap)
        else:
            if isinstance(facecolor,str):
                facecolor=[facecolor]*len(sel_contour.contours)
                
        for kk,contour in enumerate(sel_contour.contours):
            
            contour.plot_contour(ax=ax,facecolor=facecolor[kk],textcolor=textcolor)
            
        return ax
    
    def select(self,contour_keys=None):
        if contour_keys is None:
            return Contours()
        
        contour_list=[x for contour_key in contour_keys for x in self.contours if x.key==contour_key ]
        
        return Contours(key=self.key,contours=contour_list)

    def get_xy(self,contour_key):
        contour_select=self.select(contour_keys=[contour_key])
        
        xy=contour_select.contours[0].get_xy()
        
        return xy
    
    
        




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

        
        
            
    
