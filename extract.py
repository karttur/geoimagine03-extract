'''
Created on 26 May 2021

@author: thomasgumbricht
'''

# module was initially copied from mj_extractplotdata_v74.py
import numpy as np

from geoimagine.gis import kt_gis

from geoimagine.support import EaseGridNcoordToTile

def ShapelyIntersect(ptL,x,y,sampleRadius):
    from shapely import geometry
    #calculate the intersecting area of the sample plot and the image pixel
    p0 = geometry.Point(ptL[0])
    p1 = geometry.Point(ptL[1])
    p2 = geometry.Point(ptL[2])
    p3 = geometry.Point(ptL[3])  
    sampleCircle = geometry.Point(x,y).buffer(sampleRadius)
    pointList = [p0,p1,p2,p3,p0]
    cellsquare = geometry.Polygon([[p.x, p.y] for p in pointList])
    isect = cellsquare.intersection(sampleCircle)
    return isect.area

def CellFracShapely(gt, src_offset, sampleRadius, samplePt):
    originX = gt[0]
    originY = gt[3]
    pixel_width = gt[1]
    pixel_height = gt[5]
    x1, y1, xsize, ysize = src_offset
    #print 'src_offset',src_offset
    cellSampleArea = np.zeros((ysize,xsize))
    for y in range(ysize):
        for x in range(xsize):
            ulx = (x1+x)*pixel_width+originX
            uly = (y1+y)*pixel_height+originY
            lrx = (x1+x)*pixel_width+originX+pixel_width
            lry = (y1+y)*pixel_height+originY+pixel_height #pixel height is negative
            #print 'y',y,uly,lry
            ptL = ((ulx,uly),(lrx,uly),(lrx,lry),(ulx,lry))
            #print 'ptL',ptL    
            cellSampleArea[y,x] = ShapelyIntersect(ptL,samplePt[0],samplePt[1],sampleRadius)
    return cellSampleArea / cellSampleArea.sum()

def BoundingBox(gt, bbox, lins,cols):
    originX = gt[0]
    originY = gt[3]
    pixel_width = gt[1]
    pixel_height = gt[5]
    #get the cell region to sample
    x1 = int((bbox[0] - originX) / pixel_width)
    x2 = int((bbox[1] - originX) / pixel_width)
    y1 = int((bbox[3] - originY) / pixel_height)
    y2 = int((bbox[2] - originY) / pixel_height)
    xsize = x2 - x1
    ysize = y2 - y1 
    if x1 < 0:
        x1 = 0
    if x1+xsize > cols:
        xsize = cols-x1
    if y1 < 0:
        y1 = 0
    if y1+ysize > lins:
        ysize = lins-y1
    return (x1, y1, xsize+1, ysize+1)

def ExtractRasterLayer(layer,bbox):
    '''
    '''
    #print (layer.gt, bbox, layer.lins, layer.cols)
    src_offset = BoundingBox(layer.gt, bbox, layer.lins, layer.cols)
    aVal = layer.layer.ReadAsArray(*src_offset)
    return aVal,src_offset

def ExtractPlotData(layerinD, maskLayer, masknull, maskL, plotD):

    #Extract all bands in one go
    from math import sqrt,pi
    resultD = {}
    for plotid in plotD:
        plotPt = plotD[plotid]
        sampleD = {}
        sampleD['ptsrfi'] = {}
        sampleD['plotsrfi'] = {}
        sampleD['kernelsrfi'] = {}
        sampleRadius = sqrt(plotPt.area/pi)
        #sampleRadius = 25 
        samplePt = mj_gis.LatLonPtProject((plotPt.lon,plotPt.lat),maskLayer.spatialRef.proj_cs,maskLayer.srcLayer.gt)
        x,y = samplePt
        bbox = [ x, x, y, y ]
        if ExtractMaskData(maskLayer,[masknull],bbox):
            statusD = {'status':'O','neighbors':0}
            plotD[plotPt.plotid] = {'samples':{},'sceneplot':statusD}

            continue
        if ExtractMaskData(maskLayer,maskL,bbox):
            statusD = {'status':'D','neighbors':0}
            plotD[plotPt.plotid] = {'samples':{},'sceneplot':statusD}
            continue
        #Extract the central point data
        for band in layerinD:
            rasterLayer = layerinD[band]
            aVal = ExtractRasterLayer(rasterLayer,bbox)[0]
            sampleD['ptsrfi'][band] = {'srfi':int(round(aVal[0][0]))}
        #reset bbox to sample area    
        bbox = [ x-sampleRadius, x+sampleRadius, y-sampleRadius, y+sampleRadius ]
        if ExtractMaskData(maskLayer,[masknull],bbox):
            statusD = {'status':'P','neighbors':0}
            plotD[plotPt.plotid] = {'samples':{},'sceneplot':statusD}
            continue
        if ExtractMaskData(maskLayer,maskL,bbox):
            statusD = {'status':'C','neighbors':0}
            plotD[plotPt.plotid] = {'samples':{},'sceneplot':statusD}
            continue  
        for band in layerinD:
            rasterLayer = layerinD[band]            
            aVal,src_offset = ExtractRasterLayer(rasterLayer,bbox)
            x1, y1, xsize, ysize = src_offset
            if xsize + ysize > 2:
                aFrac = CellFracShapely(rasterLayer.srcLayer.gt, src_offset, sampleRadius, samplePt)
                aSum = aFrac*aVal
                sampleD['plotsrfi'][band] = {'srfi':int(round(aSum.sum()))}  
            else:
                sampleD['plotsrfi'][band] = sampleD['ptsrfi'][band]
            if aVal.size > 8:
                sampleD['kernelsrfi'][band] = {'srfi':int(round(aVal.mean())),'srfistd':int(round(aVal.std()))}
            else:
                nbk = max(1, int(sampleRadius/rasterLayer.cellsize[0])) #nbk = neighborkernel 
                bbox = [ x-nbk*rasterLayer.cellsize[0], x+nbk*rasterLayer.cellsize[0], y-nbk*rasterLayer.cellsize[1], y+nbk*rasterLayer.cellsize[1] ]

                if ExtractMaskData(maskLayer,[masknull],bbox):
                    statusD = {'status':'R','neighbors':0}
                    plotD[plotPt.plotid] = {'samples':{},'sceneplot':statusD}
                    continue
                if ExtractMaskData(maskLayer,maskL,bbox):
                    statusD = {'status':'B','neighbors':0}
                    plotD[plotPt.plotid] = {'samples':{},'sceneplot':statusD}
                    continue  
                for band in layerinD:
                    rasterLayer = layerinD[band]            
                    aVal = ExtractRasterLayer(rasterLayer,bbox)[0]
                    sampleD['kernelsrfi'][band] = {'srfi':int(round(aVal.mean())),'srfistd':int(round(aVal.std()))}
        statusD = {'status':'A','neighbors':aVal.size}
        plotD[plotPt.plotid] = {'samples':sampleD,'sceneplot':statusD}
        #resultD[plotPt.plotid]  = {}          
    return plotD

def ExtractMaskData(maskLayer,mL,bbox):
    aVal = ExtractRasterLayer(maskLayer,bbox)[0]
    if np.any(aVal == mL):
        return True
    return False
        
class SamplePlot:
    """Periodicity sets the time span, seasonality and timestep to process data for."""   
    def __init__(self,plotPt):
        self.plotid = plotPt[0]
        self.lon = plotPt[1]
        self.lat = plotPt[2]
        self.area = plotPt[3]
        
class Layer():
    def __init__(self,FPN):
        self.FPN = FPN
        self.lins, self.cols, self.projection, self.geotrans, self.ext, self.cellsize, self.celltype, self.cellnull, self.epsg, self.epsgunit = mj_gis.GetRasterInfo(self.FPN)
        self.spatialRef = mj_gis.GetRasterMetaData(self.FPN)[0]
        self.srcDS,self.srcLayer = mj_gis.RasterOpenGetFirstLayer(self.FPN,'read')
   

        
def GetStarted(layerinD,plotPtT,masknull,maskL):
    srfiLayerD = {}
    for key in layerinD:
        if 'mask' in key:
            maskFPN = layerinD[key].FPN
            masklayer = Layer(layerinD[key].FPN)
        else:
            srfiLayerD[key] = Layer(layerinD[key].FPN)
    plotD = {}
    for plotPt in plotPtT:
        plotD[plotPt[0]] = SamplePlot(plotPt)
    plotD = ExtractPlotData(srfiLayerD, masklayer, masknull,maskL, plotD)
    return plotD
    for plot in plotD:
        print (plot,plotD[plot])
    SNULLEBULLE
  
  
class ProcessExtract(): 
    '''
    '''
    
    def __init__(self, pp, session):
        '''
        '''
                
        self.session = session
                
        self.pp = pp  
        
        self.verbose = self.pp.process.verbose 
        
        self.session._SetVerbosity(self.verbose)
        
        print ('        ProcessExtract', self.pp.process.processid) 
        
        #direct to subprocess
                    
        if self.pp.process.processid.lower() == 'ExtractTilesPointList':
            
            self._ExtractTilesPointList()
        
        elif self.pp.process.processid.lower() == 'ExtractTilesPointVector':
            
            pass
        
        elif self.pp.process.processid.lower() == 'ExtractTilesLineVector':
            
            pass
        
        elif self.pp.process.processid.lower() == 'ExtractTilesPolyVector':
            
            pass
        
        else:
            
            exitstr = 'EXITING. process %s not found under ProcessExtract' %(self.pp.process.processid)
     
    
    
    def _ExtractTilesPointList(self):
        '''
        '''
        pass
    
    def _ExtractTilesPointVector(self):
        '''
        '''
        pass
    
    def _ExtractTilesLineVector(self):
        '''
        '''
        pass
    
    
    def _ExtractTilesPolyVector(self):
        '''
        '''
        pass
    
        
if __name__ == "__main__":
    
    x = -4195516
    y = -1893291
    
    # ease2n edges
    minx = miny = -9000000
    maxx = maxy = -9000000
    # ease2n extent
    xlength = ylength = 9000000*2
    # ease2n nr of tiles
    xtiles = 20
    ytiles = 20
    
    # ease2n tile for coordinate point
    ease2nxtile = xtiles * (x-minx) / xlength
    
    ease2nytile = ytiles * (y-minx) / ylength
    
    print ('ease2nxtile',ease2nxtile)
    
    print ('ease2nytile',ease2nytile)
    
    

