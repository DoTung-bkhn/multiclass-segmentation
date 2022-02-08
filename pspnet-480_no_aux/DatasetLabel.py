from collections import namedtuple
Label = namedtuple('Label',['name','id','color'])

Camvid_labels=[Label('Animal',0,(64,128,64)),
          Label('Archway',1,(192,0,128)),
          Label('Bicyclist',2,(0,128,192)),
          Label('Bridge',3,(0,128,64)),
          Label('Building',4,(128,0,0)),
          Label('Car',5,(64,0,128)),
          Label('CartLuggagePram',6,(64,0,192)),
          Label('Child',7,(192,128,64)),
          Label('Column_Pole',8,(192,192,128)),
          Label('Fence',9,(64,64,128)),
          Label('LaneMkgsDriv',10,(128,0,192)),
          Label('LaneMkgsNonDriv',11,(192,0,64)),
          Label('Misc_Text',12,(128,128,64)),
          Label('MotorcycleScooter',13,(192,0,192)),
          Label('OtherMoving',14,(128,64,64)),
          Label('ParkingBlock',15,(64,192,128)),
          Label('Pedestrian',16,(64,64,0)),
          Label('Road',17,(128,64,128)),
          Label('RoadShoulder',18,(128,128,192)),
          Label('Sidewalk',19,(0,0,192)),
          Label('SignSymbol',20,(192,128,128)),
          Label('Sky',21,(128,128,128)),
          Label('SUVPickupTruck',22,(64,128,192)),
          Label('TrafficCone',23,(0,0,64)),
          Label('TrafficLight',24,(0,64,64)),
          Label('Train',25,(192,64,128)),
          Label('Tree',26,(128,128,0)),
          Label('Truck_Bus',27,(192,128,192)),
          Label('Tunnel',28,(64,0,64)),
          Label('VegetationMisc',29,(192,192,0)),
          Label('Void',30,(0,0,0)),
          Label('Wall',31,(64,192,0))]

VOC_2012_labels=[Label('background',0,(0,0,0)),
          Label('aeroplane',1,(128,0,0)),
          Label('bicycle',2,(0,128,0)),
          Label('bird',3,(128,128,0)),
          Label('boat',4,(0,0,128)),
          Label('bottle',5,(128,0,128)),
          Label('bus',6,(0,128,128)),
          Label('car',7,(128,128,128)),
          Label('cat',8,(64,0,0)),
          Label('chair',9,(192,0,0)),
          Label('cow',10,(64,128,0)),
          Label('diningtable',11,(192,128,0)),
          Label('dog',12,(64,0,128)),
          Label('horse',13,(192,0,128)),
          Label('motorbike',14,(64,128,128)),
          Label('person',15,(192,128,128)),
          Label('pottedplant',16,(0,64,0)),
          Label('sheep',17,(128,64,0)),
          Label('sofa',18,(0,192,0)),
          Label('train',19,(128,192,0)),
          Label('tvmonitor',20,(0,64,128)),
          Label('void',21,(128,64,12))]